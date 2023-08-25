import os
import sys

import numpy as np
import scipy as sp

sys.path.append("../../../../minq/py/minq5/")  # Needed for spsolver=2
sys.path.append("../")  
from ibcdfo.pounders import pounders

BenDFO_root = "../../../../BenDFO/"
sys.path.append(BenDFO_root + "py/")  # Needed for spsolver=2
sys.path.append('../../../manifold_sampling/py/')
sys.path.append('../../../pounders/py/')
sys.path.append('../../../manifold_sampling/py/h_examples')

from sum_squared import sum_squared
from dfoxs import dfoxs
from calfun import calfun
from goombah import goombah

os.makedirs("benchmark_results", exist_ok=True)

dfo = np.loadtxt(BenDFO_root + "data/dfo.dat")

spsolver = 2  # TRSP solver
nfmax = 50
gtol = 1e-13
factor = 10

Results = {}
GAMS_options = {}
hfun = lambda F: np.sum(F**2)

for row, (nprob, n, m, factor_power) in enumerate(dfo):
    n = int(n)
    m = int(m)

    def objective(y):
        # It is possible to have python use the same objective values via
        # octave. This can be slow on some systems. To (for example)
        # test difference between matlab and python, used the following
        # line and add "from oct2py import octave" on a system with octave
        # installed.
        # out = octave.feval("calfun_wrapper", y, m, nprob, "smooth", [], 1, 1)
        out = calfun(y, m, int(nprob), "smooth", 0, vecout=True)
        assert len(out) == m, "Incorrect output dimension"
        return np.squeeze(out)

    X0 = dfoxs(n, nprob, int(factor**factor_power))
    npmax = 2 * n + 1  # Maximum number of interpolation points [2*n+1]
    L = -np.inf * np.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
    U = np.inf * np.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]
    nfs = 1
    F0 = np.zeros((1, m))
    F0[0] = objective(X0)
    xind = 0
    delta = 0.1
    printf = False

    filename = "./benchmark_results/poundersM_and_GOOMBAH_nfmax=" + str(nfmax) + "_gtol=" + str(gtol) + "_prob=" + str(row) + "_spsolver=" + str(spsolver) + ".mat"

    Results[row] = {}
    for method in [1,2]:
        if method == 1:
            [X, F, flag, xk_best] = pounders(objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xind, L, U, printf, spsolver)
            Results[row]["alg"] = "pounders4py"
        elif method == 2:
            GAMS_options["solvers"] = range(4)
            [X, F, flag, xk_best] = goombah(sum_squared, objective, nfmax, X0, L, U, GAMS_options, "linprog")
            Results[row]["alg"] = "goombah"

        evals = F.shape[0]
        h = np.zeros(evals)

        for i in range(evals):
            h[i] = hfun(F[i, :])

        Results[row]["problem"] = "problem " + str(row) + " from More/Wild"
        Results[row]["Fvec"] = F
        Results[row]["H"] = h
        Results[row]["X"] = X


np.save("Results.npy", Results)
