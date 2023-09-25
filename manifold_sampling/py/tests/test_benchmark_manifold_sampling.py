# This wrapper tests various algorithms against the Benchmark functions from the
# More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
import os

import numpy as np
import scipy as sp
import scipy.io as sio

from calfun import calfun
from dfoxs import dfoxs
from ibcdfo.manifold_sampling.manifold_sampling_primal import manifold_sampling_primal
from ibcdfo.manifold_sampling.h_examples.piecewise_quadratic import piecewise_quadratic
from ibcdfo.manifold_sampling.h_examples.pw_maximum import pw_maximum
from ibcdfo.manifold_sampling.h_examples.pw_maximum_squared import pw_maximum_squared
from ibcdfo.manifold_sampling.h_examples.pw_minimum import pw_minimum
from ibcdfo.manifold_sampling.h_examples.pw_minimum_squared import pw_minimum_squared
from ibcdfo.manifold_sampling.h_examples.quantile import quantile

if not os.path.exists("benchmark_results"):
    os.makedirs("benchmark_results")

if not os.path.exists("mpc_test_files_smaller_Q"):
    os.system("wget https://web.cels.anl.gov/~jmlarson/mpc_test_files_smaller_Q.zip")
    os.system("unzip mpc_test_files_smaller_Q.zip")

# www.mcs.anl.gov/~jlarson/mpc_test_files_smaller_Q.zip
# Defines data for censored-L1 loss h instances
C_L1_loss = np.loadtxt("mpc_test_files_smaller_Q/C_for_benchmark_probs.csv", delimiter=",")
D_L1_loss = np.loadtxt("mpc_test_files_smaller_Q/D_for_benchmark_probs.csv", delimiter=",")
# Defines data for piecewise_quadratic h instances
Qzb = sio.loadmat("mpc_test_files_smaller_Q/Q_z_and_b_for_benchmark_problems_normalized_subset.mat")

nfmax = 50
factor = 10
subprob_switch = "linprog"
dfo = np.loadtxt("dfo.dat")
filename = "./benchmark_results/manifold_sampling_py_nfmax=" + str(nfmax) + ".mat"

Results = {}
probs_to_solve = [0, 1, 6, 7, 42, 43, 44]


subprob_switch = "linprog"


hfuns = [pw_maximum_squared, pw_maximum, piecewise_quadratic, quantile, pw_minimum_squared, pw_minimum]

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

    ind = np.where((C_L1_loss[:, 0] == probs_to_solve[row]) & (C_L1_loss[:, 1] == 0))
    C = C_L1_loss[ind, 3 : m + 3]
    D = D_L1_loss[ind, 3 : m + 3]

    # Individual for piecewise_quadratic h instance
    Qs = Qzb["Q_mat"][probs_to_solve[row], 0]
    zs = Qzb["z_mat"][probs_to_solve[row], 0]
    cs = Qzb["b_mat"][probs_to_solve[row], 0]

    for i, hfun in enumerate(hfuns):
        if hfun.__name__ == "piecewise_quadratic":

            def hfun_to_pass(z, H0=None):
                return piecewise_quadratic(z, H0, Qs=Qs, zs=zs, cs=cs)

            X, F, h, xkin, flag = manifold_sampling_primal(hfun_to_pass, Ffun, x0, LB, UB, nfmax, subprob_switch)
        else:
            X, F, h, xkin, flag = manifold_sampling_primal(hfun, Ffun, x0, LB, UB, nfmax, subprob_switch)

        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(i)] = {}
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(i)]["alg"] = "Manifold sampling"
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(i)]["problem"] = ["problem " + str(probs_to_solve[row] + 1) + " from More/Wild with hfun=" + str(hfun)]
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(i)]["Fvec"] = F
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(i)]["H"] = h
        Results["MSP_" + str(probs_to_solve[row] + 1) + "_" + str(i)]["X"] = X

    sp.io.savemat(filename, Results)
