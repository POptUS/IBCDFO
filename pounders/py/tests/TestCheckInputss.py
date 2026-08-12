"""
Unit test of checkinputss()
"""

import unittest

import numpy as np
from ibcdfo.pounders import _checkinputss as checkinputss


class TestCheckInputss(unittest.TestCase):
    def setUp(self):
        self.Ffun = np.linalg.norm
        self.n = 3
        self.X_0 = np.full(self.n, 0.5, float)
        self.np_max = 2 * self.n + 1
        self.nf_max = 10
        self.g_tol = 1e-13
        self.delta_0 = 0.1
        self.nfs = 2
        self.m = 1
        self.X_init = np.vstack((0.5 * np.ones(self.n), np.zeros(self.n)))
        self.F_init = np.zeros((self.nfs, self.m))
        self.xk_in = 0
        self.Low = np.zeros(self.n)
        self.Upp = np.ones(self.n)

        self.__NOT_REAL = (None, "", {}, [1.1], {1.1})
        self.__NOT_INT = (None, "", 1.0, 1.1, {}, [1], {1})
        self.__NOT_NUMPY_ARRAY = (None, "", 1, 1.1, {}, [1], {1})
        self.__NOT_FUNCTION = (None, "", 1, 1.1, {}, [1], {1})

    def __test(self, new_args, expected_exception):
        kwargs = {
            "Ffun": self.Ffun,
            "n": self.n,
            "X_0": self.X_0,
            "np_max": self.np_max,
            "nf_max": self.nf_max,
            "g_tol": self.g_tol,
            "delta_0": self.delta_0,
            "nfs": self.nfs,
            "m": self.m,
            "X_init": self.X_init,
            "F_init": self.F_init,
            "xk_in": self.xk_in,
            "Low": self.Low,
            "Upp": self.Upp,
        }
        for key, value in new_args.items():
            kwargs[key] = value

        if expected_exception is None:
            checkinputss(**kwargs)
        else:
            with self.assertRaises(expected_exception):
                checkinputss(**kwargs)

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
        for good in [0, 1, 10, 10_000]:
            test_case["nf_max"] = min_required + good
            self.__test(test_case, None)
        for bad in range(min_required):
            test_case["nf_max"] = bad
            self.__test(test_case, ValueError)

        # Two prior evaluation
        min_required = 3
        for good in [0, 1, 10, 10_000]:
            self.__test({"nf_max": min_required + good}, None)
        for bad in range(min_required):
            self.__test({"nf_max": bad}, ValueError)

        # n+2 or more prior evaluations
        min_required = 1
        for nfs in [5, 6, 10, 10_000]:
            test_case = {"nfs": nfs, "X_init": np.full((nfs, self.n), 0.5, float), "F_init": np.zeros((nfs, self.m)), "xk_in": 0}
            for good in [1, 2, 10, 10_000]:
                test_case["nf_max"] = good
                self.__test(test_case, None)
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
        self.__test({"Low": self.X_0}, None)
        self.__test({"Upp": self.X_0}, None)

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

        # TODO: Check invalid xk_in

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
        MIN = np.finfo(float).min
        MAX = np.finfo(float).max
        EPS = np.finfo(float).eps

        for bad in self.__NOT_INT:
            self.__test({"xk_in": bad}, TypeError)

        # Invalid integer index
        for bad in [-1, self.nfs]:
            self.__test({"xk_in": bad}, ValueError)

        bad_no_priors = {"nfs": -0, "X_init": np.full((0, self.n), 0.0, float), "F_init": np.full((0, self.m), 0.0, float), "xk_in": -1}
        for bad in [-1, 1]:
            bad_no_priors["xk_in"] = bad
            self.__test(bad_no_priors, ValueError)

        # No priors provided
        test_case = {"nfs": 0, "X_init": np.full((0, self.n), np.nan, float), "F_init": np.full((0, self.m), np.nan, float), "xk_in": 0}
        self.__test(test_case, None)
        for bad in [-1, 1, 2, 5]:
            test_case["xk_in"] = bad
            self.__test(test_case, ValueError)

        # Starting point does not match given X_0
        #
        # Intentionally set other points outside bounds, which should be
        # acceptable.
        nfs = 3
        test_case = {"nfs": nfs, "X_init": None, "F_init": np.zeros((nfs, self.m)), "xk_in": -1}
        for xk_in in range(nfs):
            test_case["xk_in"] = xk_in

            i, j = sorted(set(range(nfs)).difference({xk_in}))

            # See valid initial points pass ...
            X_init = np.full((nfs, self.n), np.nan, float)
            X_init[xk_in, :] = self.X_0.copy()
            X_init[i, :] = np.full(self.n, MIN, float)
            X_init[j, :] = np.full(self.n, MAX, float)
            test_case["X_init"] = X_init
            self.__test(test_case, None)

            # and then let's make 'em fail.
            for k in range(self.n):
                X_init[xk_in, :] = self.X_0.copy()
                X_init[xk_in, k] = (1.0 + EPS) * X_init[xk_in, k]
                self.__test(test_case, ValueError)

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
