"""
Unit test of compute function
"""

import unittest

import numpy as np
from ibcdfo.pounders.checkinputss import checkinputss


class TestLotsOfFeatures(unittest.TestCase):
    def setUp(self):
        self.fun = np.linalg.norm
        self.n = 3
        self.X0 = np.vstack((0.5 * np.ones(self.n), np.zeros(self.n)))
        self.npmax = 2 * self.n + 1
        self.nfmax = 10
        self.gtol = 1e-13
        self.delta = 0.1
        self.nfs = 2
        self.m = 1
        self.F0 = np.zeros((self.nfs, 1))
        self.xkin = 0
        self.L = np.zeros(self.n)
        self.U = np.ones(self.n)

    def __testCommonFinalConditions(self, out, flag):
        if flag == "success":
            self.assertEqual(out[0], 1, "Should not have failed")
        elif flag == "fail":
            self.assertEqual(out[0], -1, "Should have failed")
        elif flag == "warn":
            self.assertEqual(out[0], 0, "Should have warned, but not failed")

        self.assertEqual(len(out), 7, "Should always have 7 outputs from checkinputss")

    def test_checkinputts0(self):
        out = checkinputss(self.fun, self.X0, self.n, self.npmax, self.nfmax, self.gtol, self.delta, self.nfs, self.m, self.F0, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "success")

    def test_checkinputts1(self):
        fun_to_fail = []
        out = checkinputss(fun_to_fail, self.X0, self.n, self.npmax, self.nfmax, self.gtol, self.delta, self.nfs, self.m, self.F0, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts2(self):
        n_to_fail = 2
        out = checkinputss(self.fun, self.X0, n_to_fail, self.npmax, self.nfmax, self.gtol, self.delta, self.nfs, self.m, self.F0, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts3(self):
        npmax_to_warn = 1
        out = checkinputss(self.fun, self.X0, self.n, npmax_to_warn, self.nfmax, self.gtol, self.delta, self.nfs, self.m, self.F0, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "warn")

    def test_checkinputts4(self):
        nfmax_to_fail = 0
        out = checkinputss(self.fun, self.X0, self.n, self.npmax, nfmax_to_fail, self.gtol, self.delta, self.nfs, self.m, self.F0, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts5(self):
        gtol_to_fail = 0
        out = checkinputss(self.fun, self.X0, self.n, self.npmax, self.nfmax, gtol_to_fail, self.delta, self.nfs, self.m, self.F0, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts6(self):
        delta_to_fail = 0
        out = checkinputss(self.fun, self.X0, self.n, self.npmax, self.nfmax, self.gtol, delta_to_fail, self.nfs, self.m, self.F0, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts7(self):
        F0_to_fail = np.zeros((self.nfs, 3 * self.nfs))
        out = checkinputss(self.fun, self.X0, self.n, self.npmax, self.nfmax, self.gtol, self.delta, self.nfs, self.m, F0_to_fail, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts8(self):
        F0_to_warn = np.zeros((3 * self.nfs, 1))
        out = checkinputss(self.fun, self.X0, self.n, self.npmax, self.nfmax, self.gtol, self.delta, self.nfs, self.m, F0_to_warn, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "warn")

    def test_checkinputts9(self):
        F0_to_fail = np.nan * self.F0
        out = checkinputss(self.fun, self.X0, self.n, self.npmax, self.nfmax, self.gtol, self.delta, self.nfs, self.m, F0_to_fail, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts10(self):
        xkin_to_fail = -1
        out = checkinputss(self.fun, self.X0, self.n, self.npmax, self.nfmax, self.gtol, self.delta, self.nfs, self.m, self.F0, xkin_to_fail, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts11(self):
        L_to_fail = np.hstack((self.L, self.L))
        out = checkinputss(self.fun, self.X0, self.n, self.npmax, self.nfmax, self.gtol, self.delta, self.nfs, self.m, self.F0, self.xkin, L_to_fail, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts12(self):
        L_to_warn = np.atleast_2d(self.L).T
        U_to_warn = np.atleast_2d(self.U).T
        out = checkinputss(self.fun, self.X0, self.n, self.npmax, self.nfmax, self.gtol, self.delta, self.nfs, self.m, self.F0, self.xkin, L_to_warn, U_to_warn)
        self.__testCommonFinalConditions(out, "warn")

    def test_checkinputts13(self):
        L_to_fail = np.zeros((2, self.n))
        U_to_fail = np.zeros((2, self.n))
        out = checkinputss(self.fun, self.X0, self.n, self.npmax, self.nfmax, self.gtol, self.delta, self.nfs, self.m, self.F0, self.xkin, L_to_fail, U_to_fail)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts14(self):
        L_to_error = self.U
        out = checkinputss(self.fun, self.X0, self.n, self.npmax, self.nfmax, self.gtol, self.delta, self.nfs, self.m, self.F0, self.xkin, L_to_error, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts15(self):
        L_to_error = 0.9 * self.U
        out = checkinputss(self.fun, self.X0, self.n, self.npmax, self.nfmax, self.gtol, self.delta, self.nfs, self.m, self.F0, self.xkin, L_to_error, self.U)
        self.__testCommonFinalConditions(out, "fail")
