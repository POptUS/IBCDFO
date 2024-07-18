# This wrapper tests various algorithms against the Benchmark functions from the
# More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
import os

import numpy as np
from calfun import calfun
from dfoxs import dfoxs
from roboboa import roboboa
import ipdb

dfo = np.loadtxt("dfo.dat")

Results = {}
probs_to_solve = np.array([6]) #np.arange(53)

for row, (nprob, n, m, factor_power) in enumerate(dfo[probs_to_solve, :]):
    n = int(n)
    m = int(m)
    LB = -np.inf * np.ones((1, n))
    UB = np.inf * np.ones((1, n))
    x0 = dfoxs(n, nprob, 10**factor_power)

    print("run: ", row)

    def funx(y):
        f, _, g, _ = calfun(y, m, int(nprob), "smooth", 0, num_outs=4)
        return f, g

    # play with these~
    b_low = -0.01 * np.ones(n)
    b_upp = 0.01 * np.ones(n)
    nfmax = 500 * n ** 2

    x, Uhat, fUhat, gUhat, chi = roboboa(funx, x0, b_low, b_upp, nfmax, funxUhat=None, Uhat0=np.zeros((1, n)))

    print("x: ", x, "Uhat: ", Uhat, "fUhat: ", fUhat, "gUhat: ", gUhat, "chi: ", chi)