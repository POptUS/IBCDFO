"""
Unit test of POUNDERS interface.
"""

import io
import numbers
import unittest

import numpy as np

from contextlib import redirect_stdout

import ibcdfo


class TestPoundersInterface(unittest.TestCase):
    def setUp(self):
        self.__THETA_STAR = np.array([2.1, -0.4, 3.4])
        x_all = np.array([-1.1, 0.2, 1.3, 2.1])
        M = np.array([(1.0, x, x**2) for x in x_all])

        def Ffun(theta):
            return M @ (theta - self.__THETA_STAR)

        # Problem
        self.Ffun = Ffun
        self.n = len(self.__THETA_STAR)
        self.m = len(x_all)
        self.Low = np.full(self.n, -2.2, float)
        self.Upp = np.full(self.n, 4.4, float)

        # Algorithm
        self.X_0 = np.full(self.n, 0.5, float)
        self.g_tol = 1.0e-13
        self.delta_0 = 0.1
        self.nf_max = 15
        self.np_max = 2 * self.n + 1

        # Prior
        X_init = np.zeros((2, self.n))
        self.nfs = X_init.shape[0]
        self.xk_in = 1
        X_init[self.xk_in, :] = self.X_0
        self.X_init = X_init
        self.F_init = np.array([Ffun(X_init[i, :]) for i in range(self.nfs)])

        self.__NOT_REAL = (None, "", {}, [1.1], {1.1})
        self.__NOT_INT = (None, "", 1.0, 1.1, {}, [1], {1})
        self.__NOT_NUMPY_ARRAY = (None, "", 1, 1.1, {}, [1], {1})
        self.__NOT_FUNCTION = (None, "", 1, 1.1, {}, [1], {1})

    def __test(self, new_args, expected_exception):
        # ----- HARDCODED VALUE
        EPS = np.finfo(float).eps
        SUCCESS_MSG = "g is sufficiently small."
        ERROR_HDR = "Error: "

        # ----- ALTER CONFIGURATION
        # These should yield a good solution
        kwargs = {
            "Ffun": self.Ffun,
            "X_0": self.X_0,
            "n": self.n,
            "nf_max": self.nf_max,
            "g_tol": self.g_tol,
            "delta_0": self.delta_0,
            "m": self.m,
            "Low": self.Low,
            "Upp": self.Upp,
            "Prior": {
                "nfs": self.nfs,
                "X_init": self.X_init,
                "F_init": self.F_init,
                "xk_in": self.xk_in,
            },
            "Options": None,
            "Model": {
                "np_max": self.np_max,
            },
        }

        for key, value in new_args.items():
            if key in ["nfs", "X_init", "F_init", "xk_in"]:
                kwargs["Prior"][key] = value
            elif key in ["np_max"]:
                kwargs["Model"][key] = value
            else:
                kwargs[key] = value

        # ----- OPTIMIZE & ERROR CHECK
        for solver in [ibcdfo.run_pounders]:
            if expected_exception is None:
                with redirect_stdout(io.StringIO()) as buffer:
                    X, F, hF, flag, xk_in = solver(**kwargs)
                self.assertEqual(buffer.getvalue().strip(), SUCCESS_MSG)

                self.assertTrue(isinstance(X, np.ndarray))
                self.assertEqual(X.ndim, 2)
                # self.assertEqual(X.shape, (self.nf_max, self.n))

                self.assertTrue(isinstance(F, np.ndarray))
                self.assertEqual(F.ndim, 2)
                # self.assertEqual(F.shape, (self.nf_max, self.m))

                self.assertTrue(isinstance(hF, np.ndarray))
                self.assertEqual(hF.ndim, 1)
                # self.assertEqual(len(hF), self.nf_max)

                self.assertTrue(isinstance(flag, numbers.Integral))
                self.assertEqual(flag, 0)

                self.assertTrue(isinstance(xk_in, numbers.Integral))
                # self.assertTrue(0 <= xk_in < self.nf_max)

                max_rel_err = np.max(np.fabs(1.0 - X[xk_in, :] / self.__THETA_STAR))
                self.assertTrue(max_rel_err <= 4.0 * EPS)
                self.assertTrue(np.max(np.fabs(F[xk_in, :])) <= 4.0 * EPS)
                self.assertTrue(np.fabs(hF[xk_in]) <= 4.0 * EPS)
            else:
                with self.assertRaises(expected_exception) as err:
                    solver(**kwargs)
                err_msg = str(err.exception)
                # print(err_msg)
                self.assertTrue(err_msg.startswith(ERROR_HDR))

    def testConfirmGoodArguments(self):
        self.__test({}, None)

    def testFfun(self):
        for bad in self.__NOT_FUNCTION:
            self.__test({"Ffun": bad}, TypeError)

    def testN(self):
        for bad in self.__NOT_INT:
            self.__test({"n": bad}, TypeError)
        for bad in [-10, -1, 0]:
            self.__test({"n": bad}, ValueError)

    def testM(self):
        for bad in self.__NOT_INT:
            self.__test({"m": bad}, TypeError)
        for bad in [-10, -1, 0]:
            self.__test({"m": bad}, ValueError)

    def testNpMax(self):
        self.assertEqual(self.n, 3)
        MIN, MAX = 4, 10

        for bad in self.__NOT_INT:
            self.__test({"np_max": bad}, TypeError)
        for good in range(MIN, MAX + 1):
            self.__test({"np_max": good}, None)
        for bad in [MIN - 1, MAX + 1]:
            self.__test({"np_max": bad}, ValueError)

    def testNfMax(self):
        self.assertEqual(self.n, 3)
        self.assertEqual(self.nfs, 2)

        for bad in self.__NOT_INT:
            self.__test({"nf_max": bad}, TypeError)

        # No prior evaluations given
        min_required = 5
        test_case = {"nfs": 0, "X_init": np.full((0, self.n), np.nan, float), "F_init": np.full((0, self.m), np.nan, float), "xk_in": 0}
        # TODO: What to do for this test?!
        # for good in [0, 1, 10, 10_000]:
        #     test_case["nf_max"] = min_required + good
        #     print(test_case["nf_max"])
        #     self.__test(test_case, None)
        for bad in range(min_required):
            test_case["nf_max"] = bad
            self.__test(test_case, ValueError)

        # Two prior evaluation
        min_required = 3
        # TODO: What to do for this test?!
        # for good in [0, 1, 10, 10_000]:
        #     self.__test({"nf_max": min_required + good}, None)
        for bad in range(min_required):
            self.__test({"nf_max": bad}, ValueError)

        # n+2 or more prior evaluations
        min_required = 1
        for nfs in [5, 6, 10, 10_000]:
            test_case = {"nfs": nfs, "X_init": np.full((nfs, self.n), 0.5, float), "F_init": np.zeros((nfs, self.m)), "xk_in": 0}
            # TODO: What to do for this test?!
            # for good in [1, 2, 10, 10_000]:
            #     test_case["nf_max"] = good
            #     self.__test(test_case, None)
            test_case["nf_max"] = 0
            self.__test(test_case, ValueError)

    def testX0Errors(self):
        EPS = np.finfo(float).eps

        # Expects 1D array ...
        for bad in self.__NOT_NUMPY_ARRAY:
            self.__test({"X_0": bad}, TypeError)
        self.__test({"X_0": np.atleast_2d(self.X_0)}, ValueError)

        # of correct length
        for bad in [self.n - 1, self.n + 1]:
            self.__test({"X_0": np.full(bad, 0.5, float)}, ValueError)

        # X_0 on bounds acceptable ...
        #
        # Since the prior evaluations are already correctly set
        # relative to X_0 it's easy and cleaner to change the bounds.
        # TODO: What to do about these?!
        # self.__test({"Low": self.X_0}, None)
        # self.__test({"Upp": self.X_0}, None)

        # outside is not
        for i in range(self.n):
            Low_to_fail = np.squeeze(self.X_0.copy())
            Low_to_fail[i] = (1.0 + EPS) * Low_to_fail[i]
            self.__test({"Low": Low_to_fail}, ValueError)

            Upp_to_fail = np.squeeze(self.X_0.copy())
            Upp_to_fail[i] = (1.0 - EPS) * Upp_to_fail[i]
            self.__test({"Upp": Upp_to_fail}, ValueError)

    def testGTol(self):
        for bad in self.__NOT_REAL:
            self.__test({"g_tol": bad}, TypeError)
        for bad in [-np.inf, -1.0, -np.finfo(float).smallest_normal, 0.0, np.nan, np.inf]:
            self.__test({"g_tol": bad}, ValueError)

    def testDelta(self):
        for bad in self.__NOT_REAL:
            self.__test({"delta_0": bad}, TypeError)
        for bad in [-np.inf, -1.0, -np.finfo(float).smallest_normal, 0.0, np.nan, np.inf]:
            self.__test({"delta_0": bad}, ValueError)

    def testNfs(self):
        for bad in self.__NOT_INT:
            self.__test({"nfs": bad}, TypeError)
        for bad in [-10, -1]:
            self.__test({"nfs": bad}, ValueError)

    def testXinitErrors(self):
        # MIN = np.finfo(float).min
        # MAX = np.finfo(float).max
        EPS = np.finfo(float).eps

        NFS = 3

        for bad in self.__NOT_NUMPY_ARRAY:
            self.__test({"X_init": bad}, TypeError)

        self.__test({"X_init": np.zeros(self.n)}, ValueError)
        self.__test({"X_init": np.zeros((self.nfs, self.n, 1))}, ValueError)

        for bad in [self.nfs - 1, self.nfs + 1]:
            self.__test({"X_init": np.zeros((bad, self.n))}, ValueError)

        for bad in [self.n - 1, self.n + 1]:
            self.__test({"X_init": np.zeros((self.nfs, bad))}, ValueError)

        for bad in [np.nan, np.inf, -np.inf]:
            self.__test({"X_init": np.full(self.X_init.shape, bad, float)}, ValueError)

        # X_init[xk_in, :] does not match given X_0
        #
        # Intentionally set other points outside bounds, which should be
        # acceptable.
        test_case = {"nfs": NFS, "X_init": None, "F_init": np.zeros((NFS, self.m)), "xk_in": -1}
        for xk_in in range(NFS):
            i, j = sorted(set(range(NFS)).difference({xk_in}))

            X_init = np.full((NFS, self.n), np.nan, float)
            test_case["xk_in"] = xk_in
            test_case["X_init"] = X_init

            # See valid initial points pass ...
            # TODO: What to do about this?
            # X_init[xk_in, :] = self.X_0.copy()
            # X_init[i, :] = np.full(self.n, MIN, float)
            # X_init[j, :] = np.full(self.n, MAX, float)
            # self.__test(test_case, None)

            # and then let's make 'em fail.
            for k in range(self.n):
                X_init[xk_in, :] = self.X_0.copy()
                X_init[xk_in, k] = (1.0 + EPS) * X_init[xk_in, k]
                self.__test(test_case, ValueError)

    def testFinitErrors(self):
        for bad in self.__NOT_NUMPY_ARRAY:
            self.__test({"F_init": bad}, TypeError)

        self.__test({"F_init": np.zeros(self.m)}, ValueError)
        self.__test({"F_init": np.zeros((self.nfs, self.m, 1))}, ValueError)

        for bad in [self.nfs - 1, self.nfs + 1]:
            self.__test({"F_init": np.zeros((bad, self.m))}, ValueError)

        for bad in [self.m - 1, self.m + 1]:
            self.__test({"F_init": np.zeros((self.nfs, bad))}, ValueError)

        for bad in [np.nan, np.inf, -np.inf]:
            self.__test({"F_init": np.full(self.F_init.shape, bad, float)}, ValueError)

    def testXKinErrors(self):
        for bad in self.__NOT_INT:
            self.__test({"xk_in": bad}, TypeError)

        # No priors provided
        test_case = {"nfs": 0, "X_init": np.full((0, self.n), np.nan, float), "F_init": np.full((0, self.m), np.nan, float), "xk_in": 0}
        self.__test(test_case, None)
        for bad in [-1, 1, 2, 5]:
            test_case["xk_in"] = bad
            self.__test(test_case, ValueError)

        # Invalid integer index with priors provided
        for bad in [-1, self.nfs]:
            self.__test({"xk_in": bad}, ValueError)

        bad_no_priors = {"nfs": -0, "X_init": np.full((0, self.n), 0.0, float), "F_init": np.full((0, self.m), 0.0, float), "xk_in": -1}
        for bad in [-1, 1]:
            bad_no_priors["xk_in"] = bad
            self.__test(bad_no_priors, ValueError)

    def testBounds(self):
        for bad in self.__NOT_NUMPY_ARRAY:
            self.__test({"Low": bad}, TypeError)
            self.__test({"Upp": bad}, TypeError)

        # Need to be 1D arrays
        self.__test({"Low": np.atleast_2d(self.Low)}, ValueError)
        self.__test({"Upp": np.atleast_2d(self.Upp)}, ValueError)

        # Incorrect lengths
        for bad in [np.zeros(self.n - 1), np.zeros(self.n + 1)]:
            self.__test({"Low": bad}, ValueError)
            self.__test({"Upp": bad}, ValueError)

        # Sensible +/-Inf values are acceptable, ...
        self.__test({"Low": np.full(self.n, -np.inf, float)}, None)
        self.__test({"Upp": np.full(self.n, np.inf, float)}, None)

        # NaN values, not so much.
        for i in range(self.n):
            Low_to_fail = self.Low.copy()
            Low_to_fail[i] = np.nan
            self.__test({"Low": Low_to_fail}, ValueError)

            Upp_to_fail = self.Upp.copy()
            Upp_to_fail[i] = np.nan
            self.__test({"Upp": Upp_to_fail}, ValueError)

        # Low >= Upp
        for i in range(self.n):
            Low_to_fail = self.Low.copy()
            Low_to_fail[i] = self.Upp[i]
            self.__test({"Low": Low_to_fail}, ValueError)
