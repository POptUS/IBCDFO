"""
Unit test of compute function
"""

from dfoxs import dfoxs
from calfun import calfun

import ibcdfo.pounders as pdrs
import numpy as np


combinemodels = pdrs.identity_combine

dfo = np.loadtxt("dfo.dat")

factor = 10

for row, (nprob, n, m, factor_power) in enumerate(dfo[10:11]):
    n = int(n)
    m = int(m)

    def Ffun(y):
        out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[0]
        # assert len(out) == m, "Incorrect output dimension"
        return np.squeeze(out)

    X0 = dfoxs(n, nprob, int(factor**factor_power))
    Low = -np.inf * np.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
    Upp = np.inf * np.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]

    npmax = 2 * n + 1
    nfmax = 500
    gtol = 10**-13
    delta = 0.1
    nfs = 1
    F0 = Ffun(X0)
    xind = 0

    printf = False
    spsolver = 1

    hfun = lambda F: np.squeeze(F)

    [X, F, flag, xkin] = pdrs.pounders(Ffun, X0, n, npmax, nfmax, gtol, delta, nfs, 1, F0, xind, Low, Upp, printf, spsolver, hfun, combinemodels)
