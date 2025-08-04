"""
In rare cases, after a subproblem solution has been evaluated,
formquad+pounders produces an interpolation set with fewer than n+1 points.
This is because the order in which points are examined for sufficient affine
independence is reverse lexicographic.

This test demonstrates that behavior, although the iteration on which the
behavior occurs can differ dramatically from system to system. 
"""

import ibcdfo.pounders as pdrs
import numpy as np
from calfun import calfun
from dfoxs import dfoxs

dfo = np.loadtxt("dfo.dat")

spsolver = 2
nf_max = 1000
g_tol = 1e-13
combinemodels = pdrs.identity_combine
hfun = lambda F: np.squeeze(F)
Opts = {"printf": 1, "spsolver": 1, "hfun": hfun, "combinemodels": combinemodels}

for row, (nprob, n, m, factor_power) in enumerate(dfo[10:11]):
    n = int(n)
    m = int(m)

    def Ffun(y):
        out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[0]
        # assert len(out) == m, "Incorrect output dimension"
        return np.squeeze(out)

    X_0 = dfoxs(n, nprob, int(10**factor_power))
    Low = -np.inf * np.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
    Upp = np.inf * np.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]
    nfs = 1
    F_init = np.zeros((1, 1))
    F_init[0] = Ffun(X_0)
    xind = 0
    delta = 0.1

    Prior = {"X_init": X_0, "F_init": F_init, "nfs": nfs, "xk_in": xind}

    Results = {}

    Prior = {"nfs": 1, "F_init": F_init, "X_init": X_0, "xk_in": xind}

    [X, F, hF, flag, xk_best] = pdrs.pounders(Ffun, X_0, n, nf_max, g_tol, delta, 1, Low, Upp, Prior=Prior, Options=Opts, Model={})

    evals = F.shape[0]

    assert flag != 1, "pounders failed"
    assert hfun(F[0]) > hfun(F[xk_best]), "No improvement found"
    assert X.shape[0] <= nf_max + nfs, "POUNDERs grew the size of X"
