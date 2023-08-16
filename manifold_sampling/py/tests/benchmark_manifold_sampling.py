# This wrapper tests various algorithms against the Benchmark functions from the
# More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
import os
import sys

import numpy as np
import scipy as sp

sys.path.append("../")
sys.path.append("../../../../BenDFO/py/")
sys.path.append("../../../../BenDFO/data/")
sys.path.append("../h_examples/")

from calfun import calfun
from dfoxs import dfoxs
from manifold_sampling_primal import manifold_sampling_primal
from pw_maximum_squared import pw_maximum_squared
from pw_minimum_squared import pw_minimum_squared

if not os.path.exists("benchmark_results"):
    os.makedirs("benchmark_results")

nfmax = 500
factor = 10
subprob_switch = "linprog"
dfo = np.loadtxt("../../../../BenDFO/data/dfo.dat")
filename = "./benchmark_results/manifold_sampling_py_nfmax=" + str(nfmax) + ".mat"

Results = {}
probs_to_solve = [0, 1, 6, 7, 42, 43, 44]
for row, (nprob, n, m, factor_power) in enumerate(dfo[probs_to_solve, :]):
    n = int(n)
    m = int(m)
    LB = -np.inf * np.ones((1, n))
    UB = np.inf * np.ones((1, n))
    x0 = dfoxs(n, nprob, factor**factor_power)

    def Ffun(y):
        out = calfun(y, m, int(nprob), "smooth", 0, vecout=True)
        assert len(out) == m, "Incorrect output dimension"
        return np.squeeze(out)

    for hfun_case in [1, 2]:
        if hfun_case == 1:
            hfun = pw_maximum_squared
        if hfun_case == 2:
            hfun = pw_minimum_squared

        X, F, h, xkin, flag = manifold_sampling_primal(hfun, Ffun, x0, LB, UB, nfmax, subprob_switch)

        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(hfun_case)] = {}
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(hfun_case)]["alg"] = "Manifold sampling"
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(hfun_case)]["problem"] = ["problem " + str(probs_to_solve[row] + 1) + " from More/Wild with hfun=" + str(hfun)]
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(hfun_case)]["Fvec"] = F
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(hfun_case)]["H"] = h
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(hfun_case)]["X"] = X

    sp.io.savemat(filename, Results)
