"""
Unit test of simple functionality of pounders routine.
"""

import copy
import unittest

import ibcdfo
import numpy as np


class TestPounders(unittest.TestCase):
    def setUp(self):
        self.__solvers = copy.deepcopy(ibcdfo.pounders.constants.TRSP_SOLVERS)

    def test_failing_objective(self):
        # Test that a NaN/Inf encountered at any point during the initial
        # point evaluation or the model-building points behaves as expected
        # in both serial and batched modes.
        def make_Ffun(fail_at_eval, bad_value, use_batched):
            n_evals = [0]

            def F_single_eval(X):
                F = X.copy()
                if n_evals[0] == fail_at_eval:
                    F[0] = bad_value
                n_evals[0] += 1
                return F

            def F_batched(X):
                F = np.full(X.shape, np.nan, float)
                for i in range(F.shape[0]):
                    F[i, :] = F_single_eval(X[i, :])
                return F

            return F_batched if use_batched else F_single_eval

        simple_solver = ibcdfo.pounders.create_trsp_solver(ibcdfo.pounders.constants.TRSP_SOLVER_SIMPLE)
        nf_max = 25
        g_tol = 1e-13
        n = 3
        m = 3

        X_0 = np.array([10.0, 20.0, 30.0])
        Low = np.full(n, -np.inf, float)
        Upp = np.full(n, np.inf, float)
        delta = 0.1

        for batched in [True, False]:
            for bad in [np.nan, np.inf, -np.inf]:
                # We don't know if the optimization will always converge within
                # the allotted budget.  If it does, we don't know how many
                # evaluations will be needed...
                for fail_at_eval in range(nf_max):
                    Ffun_to_fail = make_Ffun(fail_at_eval, bad, batched)
                    Opts = {"spsolver": simple_solver, "printf": False, "batched_Ffun": batched}
                    [X, F, hF, flag, xk_best] = ibcdfo.run_pounders(Ffun_to_fail, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts)
                    if flag == 0:
                        break
                    self.assertEqual(X.shape[0], fail_at_eval + 1, f"Earlier valid geometry point was lost. (X.shape={X.shape})")
                    self.assertTrue(np.all(np.isfinite(F[:fail_at_eval, :])))
                    self.assertTrue(np.array_equal(X[:fail_at_eval, :], F[:fail_at_eval, :]))
                    self.assertFalse(np.all(np.isfinite(F[fail_at_eval, :])))
                # But, we want to make sure that a sufficiently large number of
                # evaluations were made so that this test is sufficiently
                # stressful and meaningful.
                self.assertTrue(fail_at_eval >= 10)
                self.assertTrue(flag in {-3, 0})

    def test_basic_pounders_usage(self):
        def Ffun(x):
            return x + (x**2)

        # n [int] Dimension (number of continuous variables)
        n = 2
        # X_0 [dbl] [min(fstart,1)-by-n] Set of initial points  (zeros(1,n))
        X_0 = np.zeros((10, 2))
        X_0[0, :] = 0.5 * np.ones((1, 2))
        # nf_max [int] Maximum number of function evaluations (>n+1) (100)
        nf_max = 60
        # g_tol [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
        g_tol = 10**-13
        # delta [dbl] Positive trust region radius (.1)
        delta = 0.1
        # nfs [int] Number of function values (at X_0) known in advance (0)
        nfs = 10
        # m [int] number of residuals
        m = 2
        # F_init [dbl] [fstart-by-1] Set of known function values  ([])
        F_init = np.zeros((X_0.shape[0], m))
        # xind [int] Index of point in X_0 at which to start from (1)
        xind = 0
        # Low [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
        Low = np.zeros(n)
        # Upp [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
        Upp = np.ones(n)

        np.random.seed(1)
        F_init[0, :] = Ffun(X_0[0, :])
        for i in range(1, X_0.shape[0]):
            X_0[i, :] = X_0[0, :] + 0.2 * np.random.rand(1, 2) - 0.1
            F_init[i, :] = Ffun(X_0[i, :])

        Prior = {"X_init": X_0, "F_init": F_init, "nfs": nfs, "xk_in": xind}
        [X, F, hF, flag, xk_in] = ibcdfo.run_pounders(Ffun, X_0[xind], n, nf_max, g_tol, delta, m, Low, Upp, Model={"np_max": int(0.5 * (n + 1) * (n + 2))}, Prior=Prior)

    def test_pounders_one_output(self):
        simple_solver = ibcdfo.pounders.create_trsp_solver(ibcdfo.pounders.constants.TRSP_SOLVER_SIMPLE)

        hfun = ibcdfo.pounders.h_identity
        combinemodels = ibcdfo.pounders.combine_identity

        # Sample calling syntax for pounders
        Ffun = lambda x: np.sum(x)
        n = 16

        X_0 = np.ones(n)
        nf_max = 800
        g_tol = 10**-13
        delta = 0.1
        nfs = 1
        m = 1
        X_init = np.atleast_2d(X_0)
        F_init = np.atleast_2d(Ffun(X_0))
        xind = 0
        Low = -0.1 * np.arange(n)
        Upp = np.inf * np.ones(n)

        Opts = {"spsolver": simple_solver, "hfun": hfun, "combinemodels": combinemodels}
        Prior = {"X_init": X_init, "F_init": F_init, "nfs": nfs, "xk_in": xind}
        [X, F, hF, flag, xk_in] = ibcdfo.run_pounders(Ffun, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts, Prior=Prior)
        self.assertTrue(np.linalg.norm(X[xk_in] - Low) <= 1e-8, f"The minimum should be at the lower bounds. (X[xk_in]={X[xk_in]})")

        Ffun = lambda x: np.sum(x**2)
        Opts = {"spsolver": simple_solver, "hfun": hfun, "combinemodels": combinemodels}
        [X, F, hF, flag, xk_in] = ibcdfo.run_pounders(Ffun, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts, Prior=Prior)
        self.assertTrue(flag == -2, f"This test should terminate because mdec == 0.  (flag={flag})")

        Opts = {"spsolver": simple_solver, "hfun": hfun, "combinemodels": combinemodels, "delta_min": 1e-1}
        [X, F, hF, flag, xk_in] = ibcdfo.run_pounders(Ffun, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts, Prior=Prior)
        self.assertTrue(flag == -6, f"This test should hit the mindelta termination (flag={flag}).")

    def test_pounders_maximizing_sum_squares(self):
        # Sample calling syntax for pounders
        Ffun = lambda x: x
        n = 16

        X_0 = 0.4 * np.ones(n)  # Test giving of column vector
        nf_max = 200
        g_tol = 10**-13
        delta = 0.1
        m = n
        Low = 0.1 * np.ones(n)
        Upp = np.ones(n)

        Opts = {
            "spsolver": None,
            "hfun": ibcdfo.pounders.h_neg_leastsquares,
            "combinemodels": ibcdfo.pounders.combine_neg_leastsquares,
            "printf": 2,
        }
        Prior = {
            "X_init": np.atleast_2d(X_0.T),
            "F_init": np.atleast_2d(Ffun(X_0.T)),
            "nfs": 1,
            "xk_in": 0,
        }

        for idx in self.__solvers:
            Opts["spsolver"] = ibcdfo.pounders.create_trsp_solver(idx)

            [X, F, hF, flag, xk_in] = ibcdfo.run_pounders(Ffun, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts, Prior=Prior)

            self.assertTrue(np.linalg.norm(X[xk_in] - Upp) <= 1e-8, f"The minimum should be at the upper bounds. (X[xk_in]={X[xk_in]})")

    def test_pounders_one_dimensional(self):
        def Ffun(x):
            """
            Smooth R -> R^3 function with ||f(x)||_2 minimized at x = 0.7.
            Returns a 3-vector.
            """
            t = x.squeeze() - 0.7
            return [t, t**2, t**3]

        n = 1
        X_0 = 0.4 * np.ones(n)
        nf_max = 200
        g_tol = 10**-13
        delta = 0.1
        m = 3
        Low = 0.1 * np.ones(n)
        Upp = np.ones(n)

        Opts = {"spsolver": None}

        for idx in self.__solvers:
            Opts["spsolver"] = ibcdfo.pounders.create_trsp_solver(idx)
            [X, F, hF, flag, xk_in] = ibcdfo.run_pounders(Ffun, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts)

            self.assertTrue(np.linalg.norm(X[xk_in] - 0.7) <= 1e-8, f"The minimum should be close to 0.7. (X[xk_in]={X[xk_in]})")
