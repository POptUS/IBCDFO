"""
Unit test of compute function
"""

import os
import unittest

import ibcdfo.pounders as pdrs
import numpy as np
import scipy as sp
from calfun import calfun
from dfoxs import dfoxs


if not os.path.exists("benchmark_results"):
    os.makedirs("benchmark_results")

dfo = np.loadtxt("dfo.dat")

spsolver = 2
nf_max = 500
g_tol = 1e-13
factor = 10
printf = False
combinemodels = pdrs.identity_combine
hfun = lambda F: np.squeeze(F)
Opts = {"spsolver": 1, "hfun": hfun, "combinemodels": combinemodels}

for row, (nprob, n, m, factor_power) in enumerate(dfo[10:11]):
    n = int(n)
    m = int(m)

    def Ffun(y):
        out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[0]
        # assert len(out) == m, "Incorrect output dimension"
        return np.squeeze(out)

    X_0 = dfoxs(n, nprob, int(factor**factor_power))
    Low = -np.inf * np.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
    Upp = np.inf * np.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]
    nfs = 1
    F_init = np.zeros((1, 1))
    F_init[0] = Ffun(X_0)
    xind = 0
    delta = 0.1

    Prior = {"X_init": X_0, "F_init": F_init, "nfs": nfs, "xk_in": xind}

    Results = {}

    filename = "./benchmark_results/pounders4py_nf_max=" + str(nf_max) + "_prob=" + str(row) + "_spsolver=" + str(spsolver) + "_hfun=" + combinemodels.__name__ + ".mat"
    Prior = {"nfs": 1, "F_init": F_init, "X_init": X_0, "xk_in": xind}

    [X, F, hF, flag, xk_best] = pdrs.pounders(Ffun, X_0, n, nf_max, g_tol, delta, 1, Low, Upp, Prior=Prior, Options=Opts, Model={})

    evals = F.shape[0]

    self.assertNotEqual(flag, 1, "pounders failed")
    self.assertTrue(hfun(F[0]) > hfun(F[xk_best]), "No improvement found")
    self.assertTrue(X.shape[0] <= nf_max + nfs, "POUNDERs grew the size of X")

    if flag == 0:
        self.assertTrue(evals <= nf_max + nfs, "POUNDERs evaluated more than nf_max evaluations")
    elif flag != -4:
        self.assertTrue(evals == nf_max + nfs, "POUNDERs didn't use nf_max evaluations")

    Results["pounders4py_" + str(row) + "_" + str(hfun_cases)] = {}
    Results["pounders4py_" + str(row) + "_" + str(hfun_cases)]["alg"] = "pounders4py"
    Results["pounders4py_" + str(row) + "_" + str(hfun_cases)]["problem"] = "problem " + str(row) + " from More/Wild"
    Results["pounders4py_" + str(row) + "_" + str(hfun_cases)]["Fvec"] = F
    Results["pounders4py_" + str(row) + "_" + str(hfun_cases)]["H"] = hF
    Results["pounders4py_" + str(row) + "_" + str(hfun_cases)]["X"] = X
    # oct2py.kill_octave() # This is necessary to restart the octave instance,
    #                      # and thereby remove some caching of inside of oct2py,
    #                      # namely changing problem dimension does not
    #                      # correctly redefine calfun_wrapper

    sp.io.savemat(filename, Results)
