"""
Unit test of compute function
"""

import unittest

import ibcdfo.pounders as pdrs
import numpy as np


combinemodels = pdrs.identity_combine

# Sample calling syntax for pounders
func = lambda x: np.sum(x)
n = 16

dfo = np.loadtxt("dfo.dat")

for row, (nprob, n, m, factor_power) in enumerate(dfo[10:11]):
    n = int(n)
    m = int(m)

    def Ffun(y):
        out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[0]
        # assert len(out) == m, "Incorrect output dimension"
        return np.squeeze(out)

    Low = -np.inf * np.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
    Upp = np.inf * np.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]

    X0 = np.ones((n, 1))  # Test giving of column vector
    npmax = 2 * n + 1
    nfmax = 500
    gtol = 10**-13
    delta = 0.1
    nfs = 1
    m = 1
    F0 = func(X0)
    xind = 0
    Low = -0.1 * np.arange(n)
    Upp = np.inf * np.ones(n)
    printf = False
    spsolver = 1

    hfun = lambda F: np.squeeze(F)

    [X, F, flag, xkin] = pdrs.pounders(func, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf, spsolver, hfun, combinemodels)
