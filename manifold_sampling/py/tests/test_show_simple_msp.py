# This wrapper tests various algorithms against the Benchmark functions from the
# More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
import os

import numpy as np
from calfun import calfun
from dfoxs import dfoxs
import ibcdfo

if not os.path.exists("mpc_test_files_smaller_Q"):
    os.system("wget https://web.cels.anl.gov/~jmlarson/mpc_test_files_smaller_Q.zip")
    os.system("unzip mpc_test_files_smaller_Q.zip")

# www.mcs.anl.gov/~jlarson/mpc_test_files_smaller_Q.zip
# Defines data for censored-L1 loss h instances
C_L1_loss = np.loadtxt("mpc_test_files_smaller_Q/C_for_benchmark_probs.csv", delimiter=",")
D_L1_loss = np.loadtxt("mpc_test_files_smaller_Q/D_for_benchmark_probs.csv", delimiter=",")

dfo = np.loadtxt("dfo.dat")

Results = {}
probs_to_solve = [1]

subprob_switch = "linprog"

hfuns = [ibcdfo.manifold_sampling.h_one_norm, ibcdfo.manifold_sampling.h_censored_L1_loss]
nfmax = 150

for row, (nprob, n, m, factor_power) in enumerate(dfo[probs_to_solve, :]):
    n = int(n)
    m = int(m)
    LB = -np.inf * np.ones((1, n))
    UB = np.inf * np.ones((1, n))
    x0 = dfoxs(n, nprob, 10**factor_power)

    def Ffun(y):
        out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[1]
        assert len(out) == m, "Incorrect output dimension"
        return np.squeeze(out)

    ind = np.where((C_L1_loss[:, 0] == probs_to_solve[row] + 1) & (C_L1_loss[:, 1] == 1))
    C = C_L1_loss[ind, 3 : m + 3]
    D = D_L1_loss[ind, 3 : m + 3]

    for i, hfun in enumerate(hfuns):
        if hfun.__name__ == "h_censored_L1_loss":

            def hfun_to_pass(z, H0=None):
                return ibcdfo.manifold_sampling.h_censored_L1_loss(z, H0, C=C, D=D)

            X, F, h, xkin, flag = ibcdfo.run_MSP(hfun_to_pass, Ffun, x0, LB, UB, nfmax, subprob_switch)
        else:
            X, F, h, xkin, flag = ibcdfo.run_MSP(hfun, Ffun, x0, LB, UB, nfmax, subprob_switch)
