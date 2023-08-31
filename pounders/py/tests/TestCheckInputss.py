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

    def test_checkinputts1(self):
        fun_to_fail = []
        flag = checkinputss(fun_to_fail, self.X0, self.n, self.npmax, self.nfmax, self.gtol, self.delta, self.nfs, self.m, self.F0, self.xkin, self.L, self.U)[0]
        assert flag == -1

    def test_checkinputts2(self):
        n_to_fail = 2
        flag = checkinputss(fun, X0, n_to_fail, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U)[0]
        assert flag == -1

    def test_checkinputts3(self):
        npmax_to_warn = 1
        flag = checkinputss(fun, X0, n, npmax_to_warn, nfmax, gtol, delta, nfs, m, F0, xkin, L, U)[0]
        assert flag == 0

    def test_checkinputts4(self):
        nfmax_to_fail = 0
        flag = checkinputss(fun, X0, n, npmax, nfmax_to_fail, gtol, delta, nfs, m, F0, xkin, L, U)[0]
        assert flag == -1

    def test_checkinputts5(self):
        gtol_to_fail = 0
        flag = checkinputss(fun, X0, n, npmax, nfmax, gtol_to_fail, delta, nfs, m, F0, xkin, L, U)[0]
        assert flag == -1

    def test_checkinputts6(self):
        delta_to_fail = 0
        flag = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta_to_fail, nfs, m, F0, xkin, L, U)[0]
        assert flag == -1

    def test_checkinputts7(self):
        F0_to_fail = np.zeros((2, 7))
        flag = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0_to_fail, xkin, L, U)[0]
        assert flag == -1

    def test_checkinputts8(self):
        F0_to_warn = np.zeros((7, 1))
        flag = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0_to_warn, xkin, L, U)[0]
        assert flag == 0

    def test_checkinputts9(self):
        F0_to_fail = np.nan * F0
        flag = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0_to_fail, xkin, L, U)[0]
        assert flag == -1

    def test_checkinputts10(self):
        xkin_to_fail = -1
        flag = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin_to_fail, L, U)[0]
        assert flag == -1

    def test_checkinputts11(self):
        L_to_fail = np.hstack((L, L))
        flag = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L_to_fail, U)[0]
        assert flag == -1

    def test_checkinputts12(self):
        L_to_warn = np.atleast_2d(L).T
        U_to_warn = np.atleast_2d(U).T
        flag = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L_to_warn, U_to_warn)[0]
        assert flag == 0

    def test_checkinputts13(self):
        L_to_fail = np.zeros((2, n))
        U_to_fail = np.zeros((2, n))
        flag = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L_to_fail, U_to_fail)[0]
        assert flag == -1

    def test_checkinputts14(self):
        L_to_error = U
        flag = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L_to_error, U)[0]
        assert flag == -1

    def test_checkinputts15(self):
        L_to_error = 0.9 * U
        flag = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L_to_error, U)[0]
        assert flag == -1
