# This wrapper tests various algorithms against the Benchmark functions from the
# More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
import os
import sys

sys.path.append('./jaxnp_hash/')

import ibcdfo
import numpy as np
from calfun import calfun
from dfoxs import dfoxs

from jan_example import h_max_gamma_over_KY_jax as hfun

dfo = np.loadtxt("dfo.dat")

Results = {}
probs_to_solve = [16, 33]

subprob_switch = "linprog"

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

    X, F, h_msp, xkin, flag = ibcdfo.run_MSP(hfun, Ffun, x0, LB, UB, nfmax, subprob_switch)

    combinemodels = pdrs.identity_combine

    def unstructured_obj(x):
        maxout = hfun(Ffun(x))
        return np.squeeze(maxout[0])

    identity_hfun = lambda F: np.squeeze(F)

    nf_max = 200
    g_tol = 10**-13
    delta = 0.1

    Opts = {"spsolver": 1, "hfun": identity_hfun, "combinemodels": combinemodels}

    X, F, h_pounders, flag, xk_in = ibcdfo.run_pounders(unstructured_obj, x0, n, nf_max, g_tol, delta, 1, LB, UB, Options=Opts)
