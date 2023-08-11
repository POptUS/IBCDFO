# This wrapper tests various algorithms against the Benchmark functions from the
# More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
import numpy as np
import sys,os

sys.path.append("../")
sys.path.append("../../../../BenDFO/py/")
sys.path.append("../../../../BenDFO/data/")
sys.path.append("../h_examples/")
sys.path.append("../../../pounders/py")

from calfun import calfun
from dfoxs import dfoxs
from pw_maximum_squared import pw_maximum_squared
from pw_minimum_squared import pw_minimum_squared
from manifold_sampling_primal import manifold_sampling_primal

if not os.path.exists("benchmark_results"):
    os.makedirs("benchmark_results")

nfmax = 500
factor = 10
subprob_switch = "linprog"
dfo = np.loadtxt("../../../../BenDFO/data/dfo.dat")
filename = "./benchmark_results/manifold_samplingM_nfmax=" + str(nfmax) + ".npy"

Results = {}
for row, (nprob, n, m, factor_power) in enumerate(dfo[[0, 1, 6, 7, 42, 43, 44],:]):
    n = int(n)
    m = int(m)
    LB = -np.inf * np.ones((1, n))
    UB = np.inf * np.ones((1, n))
    x0 = dfoxs(n, nprob, factor**factor_power)

    def Ffun(y):
        out = calfun(y, m, int(nprob), "smooth", 0, vecout=True)
        assert len(out) == m, "Incorrect output dimension"
        return np.squeeze(out)

    for hfun_case in [1,2]:
        if hfun_case == 1:
            hfun = pw_maximum_squared
        if hfun_case == 2:
            hfun = pw_minimum_squared

        X, F, h, xkin, flag = manifold_sampling_primal(hfun, Ffun, x0, LB, UB, nfmax, subprob_switch)
        Results["MS-P_" + str(row) + "_" + str(hfun_case)] = {}
        Results["MS-P_" + str(row) + "_" + str(hfun_case)]["alg"] = "Manifold sampling"
        Results["MS-P_" + str(row) + "_" + str(hfun_case)]["problem"] = ["problem " + str(row) + " from More/Wild with hfun=" + str(hfun)] 
        Results["MS-P_" + str(row) + "_" + str(hfun_case)]["Fvec"] = F
        Results["MS-P_" + str(row) + "_" + str(hfun_case)]["H"] = h
        Results["MS-P_" + str(row) + "_" + str(hfun_case)]["X"] = X

    np.save(filename, "Results")
