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
        # failing_objective supports both single-point (1D) and batched (2D)
        # calls so that this test exercises every mbp_evaluator, not just the
        # default one.
        def failing_objective(X, nan_freq=0.1):
            X = np.atleast_2d(X)
            fvec = X.copy()
            for i in range(fvec.shape[0]):
                if np.random.uniform() < nan_freq:
                    fvec[i, 0] = np.nan
            return fvec

        simple_solver = ibcdfo.pounders.create_trsp_solver(ibcdfo.pounders.constants.TRSP_SOLVER_SIMPLE)
        nf_max = 1000
        g_tol = 1e-13
        n = 3
        m = 3

        X_0 = np.array([10.0, 20.0, 30.0])
        Low = np.full(n, -np.inf, float)
        Upp = np.full(n, np.inf, float)
        delta = 0.1
        printf = 1

        # This must hold no matter how the model-building points needed to
        # complete the initial interpolation set are evaluated.
        for mbp_eval in ibcdfo.pounders.constants.MBP_EVALUATORS:
            with self.subTest(mbp_eval=mbp_eval):
                np.random.seed(1)

                Opts = {"spsolver": simple_solver, "printf": printf, "mbp_evaluator": ibcdfo.pounders.create_mbp_evaluator(mbp_eval)}
                [X, F, hF, flag, xk_best] = ibcdfo.run_pounders(failing_objective, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts)
                self.assertEqual(flag, -3, f"No NaN was encountered in this test, but should have been. (mbp_eval={mbp_eval}, flag={flag})")

                Ffun_to_fail = lambda X: failing_objective(X, 1.0)
                [X, F, hF, flag, xk_best] = ibcdfo.run_pounders(Ffun_to_fail, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts)
                self.assertEqual(flag, -3, f"NaN should have been encountered on first eval. (mbp_eval={mbp_eval}, flag={flag})")

        # The dimension check on the very first evaluation happens before any
        # mbp_evaluator is ever invoked, so this case need not be parametrized.
        Ffun_to_fail = lambda x: np.hstack((x, x))
        Opts = {"spsolver": simple_solver, "printf": printf}
        [X, F, hF, flag, xk_best] = ibcdfo.run_pounders(Ffun_to_fail, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts)
        self.assertEqual(flag, -1, f"Dimension error should have occurred on first eval. (flag={flag})")

    def test_mbp_evaluator_batch_failure_preserves_earlier_points(self):
        # Test that a NaN/Inf encountered partway through a batch of
        # model-building points behaves as expected. (Earlier, valid
        # points from that same batch aren't discarded.)
        def make_indexed_nan_objective(fail_at_index):
            """
            Ffun that supports both single-point (1D) and batched (2D)
            calls, is the identity everywhere, and returns NaN in
            component 0 of exactly the fail_at_index-th point evaluated
            (0-based, counting every row of every call, in the order Ffun
            sees them).
            """
            seen = [0]

            def Ffun(X):
                was_1d = X.ndim == 1
                X = np.atleast_2d(X).copy()
                for i in range(X.shape[0]):
                    if seen[0] == fail_at_index:
                        X[i, 0] = np.nan
                    seen[0] += 1
                return X[0] if was_1d else X

            return Ffun

        simple_solver = ibcdfo.pounders.create_trsp_solver(ibcdfo.pounders.constants.TRSP_SOLVER_SIMPLE)
        n = 3
        m = 3
        X_0 = np.array([10.0, 20.0, 30.0])
        Low = np.full(n, -np.inf, float)
        Upp = np.full(n, np.inf, float)
        delta = 0.1

        for mbp_eval in ibcdfo.pounders.constants.MBP_EVALUATORS:
            with self.subTest(mbp_eval=mbp_eval):
                # Point 0 is the initial evaluation of X_0.  Points 1, 2, ...
                # are the model-building points needed to complete the first
                # interpolation set; fail on the second of these (index 2) so
                # that the first (index 1) must be preserved.
                Ffun = make_indexed_nan_objective(fail_at_index=2)
                Opts = {"spsolver": simple_solver, "mbp_evaluator": ibcdfo.pounders.create_mbp_evaluator(mbp_eval)}
                [X, F, hF, flag, xk_best] = ibcdfo.run_pounders(Ffun, X_0, n, 1000, 1e-13, delta, m, Low, Upp, Options=Opts)

                self.assertEqual(flag, -3, f"Expected a NaN failure. (mbp_eval={mbp_eval}, flag={flag})")
                self.assertEqual(X.shape[0], 3, f"Earlier valid geometry point was lost. (mbp_eval={mbp_eval}, X.shape={X.shape})")
                self.assertFalse(np.any(np.isnan(F[1])), f"Valid geometry point's F was discarded/corrupted. (mbp_eval={mbp_eval}, F[1]={F[1]})")
                self.assertTrue(np.array_equal(F[1], X[1]), f"Valid geometry point's F does not match the identity Ffun. (mbp_eval={mbp_eval}, F[1]={F[1]}, X[1]={X[1]})")

    def test_basic_pounders_usage(self):
        def vecFun(x):
            """
            Input:
                x is a NumPy array (column / row vector)
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
        F_init = np.zeros((10, 2))
        # xind [int] Index of point in X_0 at which to start from (1)
        xind = 0
        # Low [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
        Low = np.zeros(n)
        # Upp [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
        Upp = np.ones(n)

        np.random.seed(1)
        F_init[0, :] = Ffun(X_0[0, :])
        for i in range(1, 10):
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
            return np.array([t, t**2, t**3])

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
