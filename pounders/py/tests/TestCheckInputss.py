"""
Unit test of compute function
"""

import unittest

from ibcdfo.pounders.checkinputss import checkinputss
import numpy as np

fun = np.linalg.norm
n = 3
X0 = np.vstack((0.5 * np.ones(3), np.zeros(3)))
npmax = 2 * n + 1
nfmax = 10
gtol = 1e-13
delta = 0.1
nfs = 2
m = 1
F0 = np.zeros((2, 1))
xkin = 0
L = np.zeros(n)
U = np.ones(n)


class TestLotsOfFeatures(unittest.TestCase):
    def test_checkinputts1(self):
        fun_to_fail = []
        flag = checkinputss(fun_to_fail, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U)[0]
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

    # def test_checkinputts8(self):
    #     F0_to_fail = np.zeros((7,1))
    #     flag = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0_to_fail, xkin, L, U)[0]
    #     assert flag == -1

    # def test_checkinputts5(self):
    #     F0_to_fail = np.zeros(2 * m)
    #     flag = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0_to_fail, xkin, L, U)[0]
    #     assert flag == -1


if __name__ == "__main__":  # pragma: no cover
    TestLotsOfFeatures.test_checkinputts1([])
    TestLotsOfFeatures.test_checkinputts2([])
    TestLotsOfFeatures.test_checkinputts3([])
    TestLotsOfFeatures.test_checkinputts4([])
    TestLotsOfFeatures.test_checkinputts5([])
    TestLotsOfFeatures.test_checkinputts6([])
    TestLotsOfFeatures.test_checkinputts7([])
    # TestLotsOfFeatures.test_checkinputts8([])
