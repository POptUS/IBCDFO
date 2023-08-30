"""
Unit test of compute function
"""

import unittest

from ibcdfo.pounders.checkinputss import checkinputss
import numpy as np

fun = np.linalg.norm
n = 3
X0 = 0.5*np.ones(3)
npmax = 2 * n + 1
nfmax = 10
gtol = 1e-13
delta = 0.1
nfs = 1
m = 1
F0 = np.linalg.norm(X0)
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
        F0_to_fail = np.zeros(2*m)
        flag = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0_to_fail, xkin, L, U)[0]
        assert flag == -1


if __name__ == "__main__":
    TestLotsOfFeatures.test_checkinputts1([])
    TestLotsOfFeatures.test_checkinputts2([])
    TestLotsOfFeatures.test_checkinputts3([])
    TestLotsOfFeatures.test_checkinputts4([])
    # TestLotsOfFeatures.test_checkinputts5([])
