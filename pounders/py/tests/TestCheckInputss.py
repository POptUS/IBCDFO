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
        self.delta = 0.1
        self.nfs = 2
        self.m = 1
        self.X_init = np.vstack((0.5 * np.ones(self.n), np.zeros(self.n)))
        self.F_init = np.zeros((self.nfs, self.m))
        self.xk_in = 0
        self.Low = np.zeros(self.n)
        self.Upp = np.ones(self.n)

    def __test(self, new_args, flag):
        OUTPUTS = {"success": (1, "Should not have failed"), "warn": (0, "Should have warned, but not failed"), "fail": (-1, "Should have failed")}
        self.assertTrue(flag in OUTPUTS)

        kwargs = {
            "Ffun": self.Ffun,
            "n": self.n,
            "X_0": self.X_0,
            "np_max": self.np_max,
            "nf_max": self.nf_max,
            "g_tol": self.g_tol,
            "delta": self.delta,
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

        out = checkinputss(**kwargs)
        self.assertEqual(len(out), 7)
        out_flag = out[0]
        expected, err_msg = OUTPUTS[flag.lower()]
        self.assertEqual(expected, out_flag, err_msg)

    def testConfirmGoodArguments(self):
        self.__test({}, "success")

    def testFfun(self):
        self.__test({"Ffun": []}, "fail")

    def testX0Errors(self):
        # Expects 1D array ...
        self.__test({"X_0": np.atleast_2d(self.X_0)}, "fail")

        # of correct length
        for bad in [self.n - 1, self.n + 1]:
            self.__test({"X_0": np.full(bad, 0.5, float)}, "fail")

    def testNpMax(self):
        self.__test({"np_max": 1}, "warn")

    def testNfMax(self):
        self.__test({"nf_max": 0}, "fail")

    def testGTol(self):
        for bad in [-np.inf, -1.0, -np.finfo(float).smallest_normal, 0.0, np.nan, np.inf]:
            self.__test({"g_tol": bad}, "fail")

    def testDelta(self):
        for bad in [-np.inf, -1.0, -np.finfo(float).smallest_normal, 0.0, np.nan, np.inf]:
            self.__test({"delta": bad}, "fail")

    def testXinitErrors(self):
        self.__test({"X_init": np.zeros(self.n)}, "fail")
        self.__test({"X_init": np.zeros((self.nfs, self.n, 1))}, "fail")

        for bad in [self.n - 1, self.n + 1]:
            self.__test({"X_init": np.zeros((self.nfs, bad))}, "fail")

        for bad in [self.nfs - 1, self.nfs + 1]:
            self.__test({"X_init": np.zeros((bad, self.n))}, "fail")

        for bad in [np.nan, np.inf, -np.inf]:
            self.__test({"X_init": np.full(self.X_init.shape, bad, float)}, "fail")

    def testFinitErrors(self):
        self.__test({"F_init": np.zeros(self.m)}, "fail")
        self.__test({"F_init": np.zeros((self.nfs, self.m, 1))}, "fail")

        for bad in [self.m - 1, self.m + 1]:
            self.__test({"F_init": np.zeros((self.nfs, bad))}, "fail")

        for bad in [self.nfs - 1, self.nfs + 1]:
            self.__test({"F_init": np.zeros((bad, self.m))}, "fail")

        for bad in [np.nan, np.inf, -np.inf]:
            self.__test({"F_init": np.full(self.F_init.shape, bad, float)}, "fail")

    def testXKinErrors(self):
        # Strictly not an integer
        for bad in [1.0, 1.1]:
            self.__test({"xk_in": bad}, "fail")

        # Invalid integer index
        for bad in [-1, self.nfs]:
            self.__test({"xk_in": bad}, "fail")

        bad_no_priors = {"nfs": -0, "X_init": np.full((0, self.n), 0.0, float), "F_init": np.full((0, self.m), 0.0, float), "xk_in": -1}
        for bad in [-1, 1]:
            bad_no_priors["xk_in"] = bad
            self.__test(bad_no_priors, "fail")

        # TODO: Starting point doesn't match X_0

    def testBounds(self):
        EPS = np.finfo(float).eps

        # Need to be 1D arrays
        self.__test({"Low": np.atleast_2d(self.Low)}, "fail")
        self.__test({"Upp": np.atleast_2d(self.Upp)}, "fail")

        # Incorrect lengths
        for bad in [np.zeros(self.n - 1), np.zeros(self.n + 1)]:
            self.__test({"Low": bad}, "fail")
            self.__test({"Upp": bad}, "fail")

        # Sensible +/-Inf values are acceptable, ...
        self.__test({"Low": np.full(self.n, -np.inf, float)}, "success")
        self.__test({"Upp": np.full(self.n, np.inf, float)}, "success")

        # NaN values, not so much.
        for i in range(self.n):
            Low_to_fail = self.Low.copy()
            Low_to_fail[i] = np.nan
            self.__test({"Low": Low_to_fail}, "fail")

            Upp_to_fail = self.Upp.copy()
            Upp_to_fail[i] = np.nan
            self.__test({"Upp": Upp_to_fail}, "fail")

        # Low >= Upp
        # TODO: Check with +/-Inf as well.
        for i in range(self.n):
            Low_to_fail = self.Low.copy()
            Low_to_fail[i] = self.Upp[i]
            self.__test({"Low": Low_to_fail}, "fail")

        # X_0 on bounds acceptable ...
        for i in range(self.n):
            self.__test({"Low": self.X_0}, "success")
            self.__test({"Upp": self.X_0}, "success")

        # outside is not
        for i in range(self.n):
            Low_to_fail = np.squeeze(self.X_0.copy())
            Low_to_fail[i] = (1.0 + EPS) * Low_to_fail[i]
            self.__test({"Low": Low_to_fail}, "fail")

            Upp_to_fail = np.squeeze(self.X_0.copy())
            Upp_to_fail[i] = (1.0 - EPS) * Upp_to_fail[i]
            self.__test({"Upp": Upp_to_fail}, "fail")
