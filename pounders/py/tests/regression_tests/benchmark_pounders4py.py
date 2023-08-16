# This wrapper tests various algorithms against the Benchmark functions from the
# More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"

import os
import sys

import numpy as np
import scipy as sp

sys.path.append("../../../../minq/py/minq5/")  # Needed for spsolver=2
import ibcdfo.pounders as pdrs

BenDFO_root = "../../../../../BenDFO/"
sys.path.append(BenDFO_root + "py/")  # Needed for spsolver=2
from dfoxs import dfoxs
from calfun import calfun

os.makedirs("benchmark_results", exist_ok=True)
# np.seterr("raise")


def doit():
    dfo = np.loadtxt(BenDFO_root + "data/dfo.dat")

    ensure_still_solve_problems = 0
    if ensure_still_solve_problems:
        best_found = np.loadtxt("./benchmark_results/best_found.txt")
    else:
        best_found = np.nan * np.ones((53, 3))

    spsolver = 2  # TRSP solver
    nfmax = 50
    gtol = 1e-13
    factor = 10

    for row, (nprob, n, m, factor_power) in enumerate(dfo):
        n = int(n)
        m = int(m)

        def objective(y):
            # It is possible to have python use the same objective values via
            # octave. This can be slow on some systems. To (for example)
            # test difference between matlab and python, used the following
            # line and add "from oct2py import octave" on a system with octave
            # installed. 
            # out = octave.feval("calfun_wrapper", y, m, nprob, "smooth", [], 1, 1)
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
            h = np.zeros(evals)
            for i in range(evals):
                h[i] = hfun(F[i, :])

            if ensure_still_solve_problems:
                ind = np.argmin(h)
                absdiff = np.abs(h[ind] - best_found[row, hfun_cases - 1])
                if absdiff > 0:
                    reldiff = absdiff / max(abs(best_found[row, hfun_cases - 1]), abs(h[ind]))
                    if reldiff > 3e-16:
                        print("This problem didn't find the same best value anymore.", reldiff, "denominator:", max(abs(best_found[row, hfun_cases - 1]), abs(h[ind])))
                # if flag == 0:
                #     check_stationary(X[xk_best, :], L, U, BenDFO, combinemodels)
            else:
                best_found[row, hfun_cases - 1] = np.min(h)

            assert flag != 1, "pounders failed"
            assert hfun(F[0]) > hfun(F[xk_best])
            assert X.shape[0] <= nfmax + nfs, "POUNDERs grew the size of X"

            if flag == 0:
                assert evals <= nfmax + nfs, "POUNDERs evaluated more than nfmax evaluations"
            elif flag != -4:
                assert evals == nfmax + nfs, "POUNDERs didn't use nfmax evaluations"

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
            np.savetxt("./benchmark_results/best_found.txt", best_found)


if __name__ == "__main__":
    doit()
