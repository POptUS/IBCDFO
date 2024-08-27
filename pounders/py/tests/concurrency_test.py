"""
A test of pounders with concurrent evaluations 
"""

import os
import pickle
import sys
import unittest

import ibcdfo.pounders as pdrs
import numpy as np
import scipy as sp
from calfun import calfun
from dfoxs import dfoxs
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if not os.path.exists("benchmark_results"):
    os.makedirs("benchmark_results")

dfo = np.loadtxt("dfo.dat")
dfo = dfo[[0, 1, 2, 7, 18, 22, 35, 44, 49, 50], :]  # A somewhat random subset of functions

spsolver = 2
g_tol = 1e-13
factor = 10
combinemodels = pdrs.identity_combine
hfun = lambda F: np.squeeze(F)
Opts = {"spsolver": spsolver, "hfun": hfun, "combinemodels": combinemodels, "printf": False}

for row, (nprob, n, m, factor_power) in enumerate(dfo):
    par = 1
    n = int(n)
    m_for_Ffun_only = int(m)
    nf_max = 20 * (n + 1) * par

    filename = f"./benchmark_results/pounders4py_nf_max={nf_max}_prob={row}_spsolver={spsolver}_hfun=default_par={par}.mat"
    if row % size == rank and not os.path.isfile(filename):
        print(filename, flush=True)

        def Ffun(y):
            out = calfun(y, m_for_Ffun_only, int(nprob), "smooth", 0, num_outs=1)
            return np.squeeze(out)

        X_0 = dfoxs(n, int(nprob), int(factor**factor_power))
        Low = -np.inf * np.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
        Upp = np.inf * np.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]
        nfs = 1
        F_init = np.zeros((1, 1))
        F_init[0] = Ffun(X_0)
        xind = 0
        delta = 0.1

        Prior = {"X_init": X_0, "F_init": F_init, "nfs": nfs, "xk_in": xind}

        Results = {}

        Prior = {"nfs": 1, "F_init": F_init, "X_init": X_0, "xk_in": xind}

        [X, F, hF, flag, xk_best, Xtype, Group] = pdrs.pounders(Ffun, X_0, n, nf_max, g_tol, delta, 1, Low, Upp, Prior=Prior, Options=Opts, Model={}, par=par)

        evals = F.shape[0]

        assert flag != 1, "pounders failed"
        # assert hfun(F[0]) > hfun(F[xk_best]), "No improvement found"
        # assert X.shape[0] <= nf_max + nfs, "POUNDERs grew the size of X"

        # if flag == 0:
        #     assert evals <= nf_max + nfs, "POUNDERs evaluated more than nf_max evaluations"
        # elif flag != -4 and flag != -2 and flag != -5:
        #     assert evals == nf_max + nfs, "POUNDERs didn't use nf_max evaluations"

        Results["pounders4py_" + str(row)] = {}
        Results["pounders4py_" + str(row)]["alg"] = "pounders4py"
        Results["pounders4py_" + str(row)]["problem"] = "problem " + str(row) + " from More/Wild"
        Results["pounders4py_" + str(row)]["Fvec"] = F
        Results["pounders4py_" + str(row)]["H"] = hF
        Results["pounders4py_" + str(row)]["X"] = X
        Results["pounders4py_" + str(row)]["Xtype"] = Xtype
        Results["pounders4py_" + str(row)]["Group"] = Group

        sp.io.savemat(filename, Results)

        # Remove the .mat extension and save to Python data file using pickle
        filename_pkl = filename.replace(".mat", ".pkl")
        with open(filename_pkl, "wb") as file:
            pickle.dump(Results, file)
