"""
Unit test of checkinputss()
"""

import unittest

import numpy as np
from ibcdfo.pounders import _checkinputss as checkinputss


class TestLotsOfFeatures(unittest.TestCase):
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

    def __testCommonFinalConditions(self, out, flag):
        if flag == "success":
            self.assertEqual(out[0], 1, "Should not have failed")
        elif flag == "fail":
            self.assertEqual(out[0], -1, "Should have failed")
        elif flag == "warn":
            self.assertEqual(out[0], 0, "Should have warned, but not failed")

        self.assertEqual(len(out), 7, "Should always have 7 outputs from checkinputss")

    def testInternalValuesGood(self):
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "success")

    def testFfun(self):
        Ffun_to_fail = []
        out = checkinputss(Ffun_to_fail, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

    def testX0Errors(self):
        # Expects 1D array ...
        X_0_bad = np.atleast_2d(self.X_0)
        out = checkinputss(self.Ffun, X_0_bad, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

        # of correct length
        for bad in [self.n - 1, self.n + 1]:
            X_0_bad = np.full(bad, 0.5, float)
            out = checkinputss(self.Ffun, X_0_bad, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

    def testNpMax(self):
        np_max_to_warn = 1
        out = checkinputss(self.Ffun, self.X_0, self.n, np_max_to_warn, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "warn")

    def testNfMax(self):
        nf_max_to_fail = 0
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, nf_max_to_fail, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

    def testGTol(self):
        g_tol_to_fail = 0
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, g_tol_to_fail, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

    def testDelta(self):
        delta_to_fail = 0
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, delta_to_fail, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

    def testXinitErrors(self):
        X_init_to_fail = np.zeros(self.n)
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, X_init_to_fail, self.F_init, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

        X_init_to_fail = np.zeros((self.nfs, self.n, 1))
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, X_init_to_fail, self.F_init, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

        for bad in [self.n - 1, self.n + 1]:
            X_init_to_fail = np.zeros((self.nfs, bad))
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, X_init_to_fail, self.F_init, self.xk_in, self.Low, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

        for bad in [self.nfs - 1, self.nfs + 1]:
            X_init_to_fail = np.zeros((bad, self.n))
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, X_init_to_fail, self.F_init, self.xk_in, self.Low, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

        for bad in [np.nan, np.inf, -np.inf]:
            X_init_to_fail = np.full(self.X_init.shape, bad, float)
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, X_init_to_fail, self.F_init, self.xk_in, self.Low, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

    def testFinitErrors(self):
        F_init_to_fail = np.zeros(self.m)
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, F_init_to_fail, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

        F_init_to_fail = np.zeros((self.nfs, self.m, 1))
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, F_init_to_fail, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

        for bad in [self.m - 1, self.m + 1]:
            F_init_to_fail = np.zeros((self.nfs, bad))
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, F_init_to_fail, self.xk_in, self.Low, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

        for bad in [self.nfs - 1, self.nfs + 1]:
            F_init_to_fail = np.zeros((bad, self.m))
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, F_init_to_fail, self.xk_in, self.Low, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

        for bad in [np.nan, np.inf, -np.inf]:
            F_init_to_fail = np.full(self.F_init.shape, bad, float)
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, F_init_to_fail, self.xk_in, self.Low, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

    def testXKinErrors(self):
        # Strictly not an integer
        for xk_in_to_fail in [1.0, 1.1]:
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, xk_in_to_fail, self.Low, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

        # Invalid integer index
        for xk_in_to_fail in [-1, self.nfs]:
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, xk_in_to_fail, self.Low, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

        nfs_0 = 0
        X_init_0 = np.full((0, self.n), 0.0, float)
        F_init_0 = np.full((0, self.m), 0.0, float)
        for xk_in_to_fail in [-1, 1]:
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, nfs_0, self.m, X_init_0, F_init_0, xk_in_to_fail, self.Low, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

        # TODO: Starting point doesn't match X_0

    def testBounds(self):
        EPS = np.finfo(float).eps

        # Need to be 1D arrays
        Low_to_fail = np.atleast_2d(self.Low)
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, Low_to_fail, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

        Upp_to_fail = np.atleast_2d(self.Upp)
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, Upp_to_fail)
        self.__testCommonFinalConditions(out, "fail")

        # Incorrect lengths
        for Low_to_fail in [np.zeros(self.n - 1), np.zeros(self.n + 1)]:
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, Low_to_fail, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

        for Upp_to_fail in [np.ones(self.n - 1), np.ones(self.n + 1)]:
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, Upp_to_fail)
            self.__testCommonFinalConditions(out, "fail")

        # Sensible +/-Inf values are acceptable, ...
        Low_to_pass = np.full(self.n, -np.inf, float)
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, Low_to_pass, self.Upp)
        self.__testCommonFinalConditions(out, "success")

        Upp_to_pass = np.full(self.n, np.inf, float)
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, Upp_to_pass)
        self.__testCommonFinalConditions(out, "success")

        # NaN values, not so much.
        for i in range(self.n):
            Low_to_error = self.Low.copy()
            Low_to_error[i] = np.nan
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, Low_to_error, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

            Upp_to_error = self.Upp.copy()
            Upp_to_error[i] = np.nan
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, Upp_to_fail)
            self.__testCommonFinalConditions(out, "fail")

        # Low >= Upp
        # TODO: Check with +/-Inf as well.
        for i in range(self.n):
            Low_to_fail = self.Low.copy()
            Low_to_fail[i] = self.Upp[i]
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, Low_to_fail, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

        # X_0 on bounds acceptable ...
        for i in range(self.n):
            Low_to_pass = np.squeeze(self.X_0.copy())
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, Low_to_pass, self.Upp)
            self.__testCommonFinalConditions(out, "success")

            Upp_to_pass = np.squeeze(self.X_0.copy())
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, Upp_to_pass)
            self.__testCommonFinalConditions(out, "success")

        # outside is not
        for i in range(self.n):
            Low_to_fail = np.squeeze(self.X_0.copy())
            Low_to_fail[i] = (1.0 + EPS) * Low_to_fail[i]
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, Low_to_fail, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

            Upp_to_fail = np.squeeze(self.X_0.copy())
            Upp_to_fail[i] = (1.0 - EPS) * Upp_to_fail[i]
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, Upp_to_fail)
            self.__testCommonFinalConditions(out, "fail")
