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


import jax

jax.config.update("jax_enable_x64", True)

if not os.path.exists("benchmark_results"):
    os.makedirs("benchmark_results")

dfo = np.loadtxt("dfo.dat")

spsolver = 2
nf_max = 40
# nf_max = 2000
g_tol = 1e-5
factor = 10

def hfun_j(z):
    res = np.sum(z**2)
    return res


@jax.jit
def hfun_d(z, zd):
    resd = jax.jvp(hfun_j, (z,), (zd,))
    return resd


@jax.jit
def hfun_dd(z, zd, zdt, zdd):
    _, resdd = jax.jvp(hfun_d, (z, zd), (zdt, zdd))
    return resdd


def G_combine(Cres, Gres):
    n, m = Gres.shape
    G = np.zeros(n)
    for i in range(n):
        _, G[i] = hfun_d(Cres, Gres[i, :])
    return G


def H_combine(Cres, Gres, Hres):
    n, _, m = Hres.shape
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            _, H[i, j] = hfun_dd(Cres, Gres[i, :], Gres[j, :], Hres[i, j, :])
    return H


def combinemodels_jax(Cres, Gres, Hres):
    return G_combine(Cres, Gres), H_combine(Cres, Gres, Hres)

for row, (nprob, n, m, factor_power) in enumerate(dfo):
    if row != 22:
        continue

    n = int(n)
    m = int(m)

    def Ffun(y):
        # It is possible to have python use the same Ffun values via
        # octave. This can be slow on some systems. To (for example)
        # test difference between matlab and python, used the following
        # line and add "from oct2py import octave" on a system with octave
        # installed.
        # out = octave.feval("calfun_wrapper", y, m, nprob, "smooth", [], 1, 1)
        out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[1]
        assert len(out) == m, "Incorrect output dimension"
        return np.squeeze(out)

    X_0 = dfoxs(n, nprob, int(factor**factor_power))
    Low = -np.inf * np.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
    Upp = np.inf * np.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]
    nfs = 1
    F_init = np.zeros((1, m))
    F_init[0] = Ffun(X_0)
    xind = 0
    delta = 0.1

    printf = True

    for hfun_cases in range(2):
        Results = {}
        if hfun_cases == 0:
            hfun = lambda F: np.sum(F**2)
            combinemodels = pdrs.leastsquares
        elif hfun_cases == 1:
            hfun = hfun_j 
            combinemodels = combinemodels_jax


        filename = "./benchmark_results/pounders4py_nf_max=" + str(nf_max) + "_prob=" + str(row) + "_spsolver=" + str(spsolver) + "_hfun=" + combinemodels.__name__ + ".mat"
        Opts = {"printf": printf, "spsolver": spsolver, "hfun": hfun, "combinemodels": combinemodels}
        Prior = {"nfs": 1, "F_init": F_init, "X_init": X_0, "xk_in": xind}

        [X, F, hF, flag, xk_best] = pdrs.pounders(Ffun, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Prior=Prior, Options=Opts, Model={})


        if hfun_cases == 0:
            evals_ibcdfo = F.shape[0]
        elif hfun_cases == 1:
            evals_jax = F.shape[0]

    if evals_ibcdfo != evals_jax:
        print(f"For problem {row}, IBCDFO logic uses {evals_ibcdfo} evals. Jax logic uses {evals_jax} evals.")
    
