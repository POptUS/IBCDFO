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
        self.__test({"Ffun": []}, TypeError)

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
        for bad in self.__NOT_INT:
            self.__test({"np_max": bad}, TypeError)
        # TODO: Test on both sides of interval
        self.__test({"np_max": 1}, ValueError)

    def testNfMax(self):
        for bad in self.__NOT_INT:
            self.__test({"nf_max": bad}, TypeError)
        # TODO: Test on both sides of interval

    def testX0Errors(self):
        # Expects 1D array ...
        for bad in self.__NOT_NUMPY_ARRAY:
            self.__test({"X_0": bad}, TypeError)
        self.__test({"X_0": np.atleast_2d(self.X_0)}, ValueError)

        # of correct length
        for bad in [self.n - 1, self.n + 1]:
            self.__test({"X_0": np.full(bad, 0.5, float)}, ValueError)

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

        for bad in [self.n - 1, self.n + 1]:
            self.__test({"X_init": np.zeros((self.nfs, bad))}, ValueError)

        for bad in [self.nfs - 1, self.nfs + 1]:
            self.__test({"X_init": np.zeros((bad, self.n))}, ValueError)

        for bad in [np.nan, np.inf, -np.inf]:
            self.__test({"X_init": np.full(self.X_init.shape, bad, float)}, ValueError)

    def testFinitErrors(self):
        for bad in self.__NOT_NUMPY_ARRAY:
            self.__test({"F_init": bad}, TypeError)

        self.__test({"F_init": np.zeros(self.m)}, ValueError)
        self.__test({"F_init": np.zeros((self.nfs, self.m, 1))}, ValueError)

        for bad in [self.m - 1, self.m + 1]:
            self.__test({"F_init": np.zeros((self.nfs, bad))}, ValueError)

        for bad in [self.nfs - 1, self.nfs + 1]:
            self.__test({"F_init": np.zeros((bad, self.m))}, ValueError)

        for bad in [np.nan, np.inf, -np.inf]:
            self.__test({"F_init": np.full(self.F_init.shape, bad, float)}, ValueError)

    def testXKinErrors(self):
        for bad in self.__NOT_INT:
            self.__test({"xk_in": bad}, TypeError)

        # Invalid integer index
        for bad in [-1, self.nfs]:
            self.__test({"xk_in": bad}, ValueError)

        bad_no_priors = {"nfs": -0, "X_init": np.full((0, self.n), 0.0, float), "F_init": np.full((0, self.m), 0.0, float), "xk_in": -1}
        for bad in [-1, 1]:
            bad_no_priors["xk_in"] = bad
            self.__test(bad_no_priors, ValueError)

    def testBounds(self):
        EPS = np.finfo(float).eps

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
        # TODO: Check with +/-Inf as well.
        for i in range(self.n):
            Low_to_fail = self.Low.copy()
            Low_to_fail[i] = self.Upp[i]
            self.__test({"Low": Low_to_fail}, ValueError)

        # X_0 on bounds acceptable ...
        for i in range(self.n):
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
