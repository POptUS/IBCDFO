# This wrapper tests various algorithms against the Benchmark functions from the
# More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"

import os
import sys

import numpy as np
import scipy as sp
from oct2py import octave

sys.path.append("../../../../minq/py/minq5/")  # Needed for spsolver=2
import ibcdfo.pounders as pdrs

sys.path.append("./cutest/")
from dfoxs import dfoxs
from calfun import calfun

os.makedirs("benchmark_results", exist_ok=True)
# np.seterr("raise")


def doit():
    dfo = np.loadtxt("./cutest/dfo.txt")

    ensure_still_solve_problems = 0
    if ensure_still_solve_problems:
        solved = np.loadtxt("./benchmark_results/solved.txt")  # A 0-1 matrix with 1 when problem was previously solved.
    else:
        solved = np.zeros((53, 3))

    spsolver = 2  # TRSP solver
    nfmax = 30
    gtol = 1e-13
    factor = 10

    for row, (nprob, n, m, factor_power) in enumerate(dfo):
        n = int(n)
        m = int(m)

        def objective(y):
            out = calfun(y, m, int(nprob), "smooth", 0, vecout=True)
            assert len(out) == m, "Incorrect output dimension"
            return np.squeeze(out)

        X0 = dfoxs(n, nprob, int(factor**factor_power))
        npmax = 2 * n + 1  # Maximum number of interpolation points [2*n+1]
        L = -np.inf * np.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
        U = np.inf * np.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]
        nfs = 1
        F0 = np.zeros((1, m))
        F0[0] = objective(X0)
        xind = 0
        delta = 0.1
        printf = False
        for hfun_cases in range(1, 4):
            Results = {}
            if hfun_cases == 1:
                hfun = lambda F: np.sum(F**2)
                combinemodels = pdrs.leastsquares
            elif hfun_cases == 2:
                alpha = 0  # If changed here, also needs to be adjusted in squared_diff_from_mean.py
                hfun = lambda F: np.sum((F - 1 / len(F) * np.sum(F)) ** 2) - alpha * (1 / len(F) * np.sum(F)) ** 2
                combinemodels = pdrs.squared_diff_from_mean
            elif hfun_cases == 3:
                if m != 3:  # Emittance is only defined for the case when m == 3
                    continue
                hfun = pdrs.emittance_h
                combinemodels = pdrs.emittance_combine

            filename = "./benchmark_results/pounders4py_nfmax=" + str(nfmax) + "_gtol=" + str(gtol) + "_prob=" + str(row) + "_spsolver=" + str(spsolver) + "_hfun=" + combinemodels.__name__ + ".mat"

            [X, F, flag, xk_best] = pdrs.pounders(objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xind, L, U, printf, spsolver, hfun, combinemodels)

            evals = F.shape[0]

            if ensure_still_solve_problems:
                if solved[row, hfun_cases - 1] == 1:
                    assert flag == 0, "This problem was previously solved but it's anymore."
                    check_stationary(X[xk_best, :], L, U, BenDFO, combinemodels)
            else:
                if flag == 0:
                    solved[row, hfun_cases - 1] = xk_best + 1

            assert flag != 1, "pounders failed"
            assert hfun(F[0]) > hfun(F[xk_best])
            assert X.shape[0] <= nfmax + nfs, "POUNDERs grew the size of X"

            if flag == 0:
                assert evals <= nfmax + nfs, "POUNDERs evaluated more than nfmax evaluations"
            elif flag != -4:
                assert evals == nfmax + nfs, "POUNDERs didn't use nfmax evaluations"

            h = np.zeros(evals)
            for i in range(evals):
                h[i] = hfun(F[i, :])

            Results["pounders4py_" + str(row) + "_" + str(hfun_cases)] = {}
            Results["pounders4py_" + str(row) + "_" + str(hfun_cases)]["alg"] = "pounders4py"
            Results["pounders4py_" + str(row) + "_" + str(hfun_cases)]["problem"] = "problem " + str(row) + " from More/Wild"
            Results["pounders4py_" + str(row) + "_" + str(hfun_cases)]["Fvec"] = F
            Results["pounders4py_" + str(row) + "_" + str(hfun_cases)]["H"] = h
            Results["pounders4py_" + str(row) + "_" + str(hfun_cases)]["X"] = X
            # oct2py.kill_octave() # This is necessary to restart the octave instance,
            #                      # and thereby remove some caching of inside of oct2py,
            #                      # namely changing problem dimension does not
            #                      # correctly redefine calfun_wrapper

            sp.io.savemat(filename, Results)

        if not ensure_still_solve_problems:
            np.savetxt("./benchmark_results/solved.txt", solved.astype(int), fmt="%s", delimiter=",")


if __name__ == "__main__":
    doit()
