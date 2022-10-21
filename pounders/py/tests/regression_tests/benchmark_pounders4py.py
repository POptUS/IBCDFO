# This wrapper tests various algorithms against the Benchmark functions from the
# More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"

import sys
import os
from oct2py import octave
import numpy as np
import scipy as sp
from mpi4py import MPI

sys.path.append("../../")
from pounders import pounders
import general_h_funs

os.makedirs("benchmark_results", exist_ok=True)
np.seterr("raise")


def doit():
    bendfo_root = "../../../../../BenDFO/"

    probs = np.loadtxt(bendfo_root + "/data/dfo.dat")
    octave.addpath(bendfo_root + "/m/")

    probtype = "smooth"

    nfmax = int(1000)
    gtol = 1e-13

    factor = 10

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    row = 0
    Results = {}
    for nprob, n, m, ns in probs:
        row += 1

        if row % size != rank:
            continue

        # Choose your solver:  # Currently set in pounders
        # spsolver = 1
        spsolver = 1

        for hfun_cases in range(1, 4):
            if hfun_cases == 1:
                hfun = lambda F: np.sum(F**2)
                combinemodels = general_h_funs.leastsquares
            elif hfun_cases == 2:
                alpha = 0  # If changed here, also needs to be adjusted in squared_diff_from_mean.py
                hfun = lambda F: np.sum((F - 1 / len(F) * np.sum(F)) ** 2) - alpha * (1 / len(F) * np.sum(F)) ** 2
                combinemodels = general_h_funs.squared_diff_from_mean
            elif hfun_cases == 3:
                if m != 3:  # Emittance is only defined for the case when m == 3
                    continue
                hfun = general_h_funs.emittance_h
                combinemodels = general_h_funs.emittance_combine

            filename = "./benchmark_results/pounders4py_nfmax=" + str(nfmax) + "_gtol=" + str(gtol) + "_prob=" + str(row) + "_spsolver=" + str(spsolver) + "_hfun=" + combinemodels.__name__ + ".mat"
            if os.path.isfile(filename):
                Old = sp.io.loadmat(filename)
                re_check = True
            else:
                re_check = False

            print(row, hfun_cases, flush=True)
            n = int(n)
            m = int(m)
            X0 = octave.dfoxs(float(n), nprob, factor**ns).T

            delta = 0.1
            mpmax = 2 * n + 1  # Maximum number of interpolation points [2*n+1]
            Low = -np.inf * np.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
            Upp = np.inf * np.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]

            printf = False

            def calfun(y):
                out = octave.feval("calfun_wrapper", y, m, nprob, probtype, [], 1, 1)
                assert len(out) == m, "Incorrect output dimension"
                return np.squeeze(out)

            F0 = np.zeros((1, m))
            F0[0] = calfun(X0)
            nfs = 1
            xind = 0

            [XO, FO, flagO, xkinO] = pounders(calfun, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf, spsolver, hfun, combinemodels)

            assert flagO != 1, "pounders crashed"

            evals = FO.shape[0]
            h = np.zeros(evals)

            for i in range(evals):
                h[i] = hfun(FO[i, :])

            # if re_check:
            #     assert np.all(Old["pounders4py" + str(row)]["Fvec"][0, 0] == FO), "Different min found"
            #     print(row, " passed")

            Results["pounders4py_" + str(row) + '_' + str(hfun_cases)] = {}
            Results["pounders4py_" + str(row) + '_' + str(hfun_cases)]["alg"] = "pounders4py"
            Results["pounders4py_" + str(row) + '_' + str(hfun_cases)]["problem"] = "problem " + str(row) + " from More/Wild"
            Results["pounders4py_" + str(row) + '_' + str(hfun_cases)]["Fvec"] = FO
            Results["pounders4py_" + str(row) + '_' + str(hfun_cases)]["H"] = h
            Results["pounders4py_" + str(row) + '_' + str(hfun_cases)]["X"] = XO
            # oct2py.kill_octave() # This is necessary to restart the octave instance,
            #                      # and thereby remove some caching of inside of oct2py,
            #                      # namely changing problem dimension does not
            #                      # correctly redefine calfun_wrapper

            sp.io.savemat(filename, Results)


if __name__ == "__main__":
    doit()
