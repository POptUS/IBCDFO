"""
Unit test of compute function
"""

import unittest

import numpy as np
from ibcdfo.pounders.checkinputss import checkinputss


class TestLotsOfFeatures(unittest.TestCase):
    def setUp(self):
        self.Ffun = np.linalg.norm
        self.n = 3
        self.X_0 = 0.5 * np.ones(self.n)
        self.np_max = 2 * self.n + 1
        self.nf_max = 10
        self.g_tol = 1e-13
        self.delta = 0.1
        self.nfs = 2
        self.m = 1
        self.X_init = np.vstack((0.5 * np.ones(self.n), np.zeros(self.n)))
        self.F_init = np.zeros((self.nfs, self.m))
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
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "success")

    def test_checkinputts1(self):
        Ffun_to_fail = []
        out = checkinputss(Ffun_to_fail, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts2(self):
        n_to_fail = 2
        with self.assertRaises(AssertionError):
            out = checkinputss(self.Ffun, self.X_0, n_to_fail, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xkin, self.L, self.U)

    def test_checkinputts3(self):
        np_max_to_warn = 1
        out = checkinputss(self.Ffun, self.X_0, self.n, np_max_to_warn, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "warn")

    def test_checkinputts4(self):
        nf_max_to_fail = 0
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, nf_max_to_fail, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts5(self):
        g_tol_to_fail = 0
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, g_tol_to_fail, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts6(self):
        delta_to_fail = 0
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, delta_to_fail, self.nfs, self.m, self.X_init, self.F_init, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts7(self):
        F_init_to_fail = np.zeros((self.nfs, 3 * self.nfs))
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, F_init_to_fail, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts8(self):
        F_init_to_error = np.zeros((3 * self.nfs, 1))
        with self.assertRaises(AssertionError):
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, F_init_to_error, self.xkin, self.L, self.U)

    def test_checkinputts9(self):
        F_init_to_fail = np.nan * self.F_init
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, F_init_to_fail, self.xkin, self.L, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts10(self):
        xkin_to_fail = -1
        with self.assertRaises(AssertionError):
            out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, xkin_to_fail, self.L, self.U)

    def test_checkinputts11(self):
        L_to_fail = np.hstack((self.L, self.L))
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xkin, L_to_fail, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts12(self):
        L_to_warn = np.atleast_2d(self.L).T
        U_to_warn = np.atleast_2d(self.U).T
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xkin, L_to_warn, U_to_warn)
        self.__testCommonFinalConditions(out, "warn")

    def test_checkinputts13(self):
        L_to_fail = np.zeros((2, self.n))
        U_to_fail = np.zeros((2, self.n))
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xkin, L_to_fail, U_to_fail)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts14(self):
        L_to_error = self.U
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xkin, L_to_error, self.U)
        self.__testCommonFinalConditions(out, "fail")

    def test_checkinputts15(self):
        L_to_error = 0.9 * self.U
        out = checkinputss(self.Ffun, self.X_0, self.n, self.np_max, self.nf_max, self.g_tol, self.delta, self.nfs, self.m, self.X_init, self.F_init, self.xkin, L_to_error, self.U)
        self.__testCommonFinalConditions(out, "fail")
