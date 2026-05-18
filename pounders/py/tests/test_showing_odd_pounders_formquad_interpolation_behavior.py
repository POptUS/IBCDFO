"""
In rare cases, after a subproblem solution has been evaluated,
formquad+pounders produces an interpolation set with fewer than n+1 points.
This is because the order in which points are examined for sufficient affine
independence is reverse lexicographic.

This test demonstrates that behavior, although the iteration on which the
behavior occurs can differ dramatically from system to system.
"""

import ibcdfo
import numpy as np
from calfun import calfun
from dfoxs import dfoxs

dfo = np.loadtxt("dfo.dat")

spsolver = 2
nf_max = 1000
g_tol = 1e-13
combinemodels = ibcdfo.pounders.combine_identity
hfun = ibcdfo.pounders.h_identity
crappy_trsp = ibcdfo.pounders.create_trsp_solver(ibcdfo.pounders.CRAPPY_TRSP)
Opts = {"printf": 1, "spsolver": crappy_trsp, "hfun": hfun, "combinemodels": combinemodels}

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
    delta = 0.1

    [X, F, hF, flag, xk_best] = ibcdfo.pounders.run_expert_mode(Ffun, X_0, n, nf_max, g_tol, delta, 1, Low, Upp, Options=Opts)

    evals = F.shape[0]

    assert flag != 1, "pounders failed"
    assert hfun(F[0]) > hfun(F[xk_best]), "No improvement found"
    assert X.shape[0] <= nf_max + 1, "POUNDERs grew the size of X"
