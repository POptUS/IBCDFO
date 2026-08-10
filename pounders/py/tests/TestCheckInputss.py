"""
Unit test of compute function
"""

import unittest

import numpy as np
from ibcdfo.pounders import _checkinputss as checkinputss


class TestLotsOfFeatures(unittest.TestCase):
    def setUp(self):
        self.Ffun = np.linalg.norm
        self.n = 3
        self.X_0 = np.full((1, self.n), 0.5, float)
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

    def test_checkinputts1(self):
        Ffun_to_fail = []
        out = checkinputss(Ffun_to_fail, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

    def testX0Errors(self):
        # Expects 2D row vector
        for bad in [self.n - 1, self.n, self.n + 1]:
            X_0_bad = np.full(bad, 0.5, float)
            out = checkinputss(self.Ffun, X_0_bad, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

            X_0_bad = np.full((bad, 1), 0.5, float)
            out = checkinputss(self.Ffun, X_0_bad, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

        # 2D row vector must have correct length
        for bad in [self.n - 1, self.n + 1]:
            X_0_bad = np.full((1, bad), 0.5, float)
            out = checkinputss(self.Ffun, X_0_bad, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
            self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts3(self):
        np_max_to_warn = 1
        out = checkinputss(self.Ffun, self.X_0, self.n, np_max_to_warn, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "warn")

    def test_checkinputts4(self):
        nf_max_to_fail = 0
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, nf_max_to_fail, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts5(self):
        g_tol_to_fail = 0
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, g_tol_to_fail, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, self.Low, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts6(self):
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

        # Starting point doesn't match X_0

    def test_checkinputts11(self):
        Low_to_fail = np.hstack((self.Low, self.Low))
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, Low_to_fail, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts12(self):
        Low_to_warn = np.atleast_2d(self.Low).T
        Upp_to_warn = np.atleast_2d(self.Upp).T
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, Low_to_warn, Upp_to_warn)
        self.__testCommonFinalConditions(out, "warn")

    def test_checkinputts13(self):
        Low_to_fail = np.zeros((2, self.n))
        Upp_to_fail = np.zeros((2, self.n))
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, Low_to_fail, Upp_to_fail)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts14(self):
        Low_to_error = self.Upp
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, Low_to_error, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts15(self):
        Low_to_error = 0.9 * self.Upp
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, Low_to_error, self.Upp)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts16(self):
        Low_to_error = self.Low
        Low_to_error[0] = np.nan
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xk_in, Low_to_error, self.Upp)
        self.__testCommonFinalConditions(out, "fail")
