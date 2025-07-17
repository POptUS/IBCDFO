# This wrapper tests various algorithms against the Benchmark functions from the
# More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
import os
import sys
sys.path.insert(0, "/Users/wanpingd/Downloads/codeDFO/")
# /Users/wanpingd/Downloads/codeDFO/test.py
import numpy as np
import scipy as sp
from BenDFO.py.calfun import calfun
from BenDFO.py.dfoxs import dfoxs
from ibcdfo.manifold_sampling.h_examples import one_norm, pw_maximum_squared
from ibcdfo.manifold_sampling.manifold_sampling_primal_with_gradients import manifold_sampling_primal_with_gradients

if not os.path.exists("msp_benchmark_results"):
    os.makedirs("msp_benchmark_results")

dfo = np.loadtxt("BenDFO/data/dfo.dat")

Results = {}
probs_to_solve = [52] #np.arange(53)

subprob_switch = "linprog"

# H_selection = "overapproximating_hessian"
# H_selection = "sr1_hessian"
# H_selection = "bfgs_hessian"
# H_selection = "bfgs_hessian_local"
# H_selection = "sr1_hessian_local"
H_selection = "min_norm_interpolation_hessian"
# H_selection = "zero_hessian"

hfuns = [one_norm]

for row, (nprob, n, m, factor_power) in enumerate(dfo[probs_to_solve, :]):
    n = int(n)
    m = int(m)
    nfmax = 50 * n
    LB = -np.inf * np.ones((1, n))
    UB = np.inf * np.ones((1, n))
    x0 = dfoxs(n, nprob, 10**factor_power)

    print("run: ", row)

    def Ffun(y):
        _, fvec, _, J = calfun(y, m, int(nprob), "smooth", 0, num_outs=4)
        J = J.T
        assert len(fvec) == m, "Incorrect fvec dimension"
        assert J.shape[0] == m, "Incorrect Jacobian dimensions"
        assert J.shape[1] == n, "Incorrect Jacobian dimemsions"
        return np.squeeze(fvec), J

    for i, hfun in enumerate(hfuns):
        X, F, Grad, h, nf, xkin, flag = manifold_sampling_primal_with_gradients(hfun, Ffun, x0, LB, UB, nfmax, subprob_switch, H_selection, printf=True)

        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(i)] = {}
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(i)]["alg"] = "Manifold sampling"
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(i)]["problem"] = ["problem " + str(probs_to_solve[row] + 1) + " from More/Wild with hfun=" + str(hfun)]
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(i)]["Fvec"] = F
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(i)]["H"] = h
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(i)]["X"] = X

    sp.io.savemat("./msp_benchmark_results/manifold_sampling_py_with_gradients.mat", Results)
