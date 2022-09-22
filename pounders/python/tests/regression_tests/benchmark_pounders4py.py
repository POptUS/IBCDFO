# This wrapper tests various algorithms against the Benchmark functions from the
# More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"

import sys
import os
from oct2py import octave
import numpy as np
import scipy as sp
from scipy import io
from mpi4py import MPI

sys.path.append("../../")
from pounders import pounders

os.makedirs("benchmark_results", exist_ok=True)
np.seterr("raise")


def doit():

    probs = np.loadtxt("../../../../../BenDFO/data/dfo.dat")
    octave.addpath("../../../../../BenDFO/m")

    probtype = "smooth"

    nfmax = int(100)
    gtol = 1e-13

    factor = 10

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    row = 0
    for nprob, n, m, ns in probs:
        row += 1

        if row % size != rank:
            continue

        # Choose your solver:  # Currently set in pounders
        # spsolver = 1
        spsolver = 1

        filename = "./benchmark_results/pounders4py_nfmax=" + str(nfmax) + "_gtol=" + str(gtol) + "_" + probtype + "_prob=" + str(row) + "_spsolver=" + str(spsolver) + ".mat"
        if os.path.isfile(filename):
            Old = sp.io.loadmat(filename)
            re_check = True
        else:
            re_check = False

        print(row, flush=True)
        Results = {}
        n = int(n)
        m = int(m)
        X0 = octave.dfoxs(float(n), nprob, factor**ns).T

        delta = 0.1
        mpmax = 2 * n + 1  # Maximum number of interpolation points [2*n+1]
        Low = -np.inf * np.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1,n)]
        Upp = np.inf * np.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1,n)]

        printf = False

        def calfun(y):
            out = octave.feval("calfun_wrapper", y, m, nprob, probtype, [], 1, 1)
            assert len(out) == m, "Incorrect output dimension"
            return np.squeeze(out)

        F0 = np.zeros((1,m))
        F0[0] = calfun(X0)
        nfs = 1
        xind = 0

        [XO, FO, flagO, xkinO] = pounders(calfun, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf, spsolver)

        if re_check:
            assert np.all(Old["pounders4py" + str(row)]["Fvec"][0, 0] == FO), "Different min found"
            print(row, " passed")

        Results["pounders4py" + str(row)] = {}
        Results["pounders4py" + str(row)]["alg"] = "pounders4py"
        Results["pounders4py" + str(row)]["problem"] = "problem " + str(row) + " from More/Wild"
        Results["pounders4py" + str(row)]["Fvec"] = FO
        Results["pounders4py" + str(row)]["H"] = np.sum(FO**2, axis=1)
        Results["pounders4py" + str(row)]["X"] = XO
        # oct2py.kill_octave() # This is necessary to restart the octave instance,
        #                      # and thereby remove some caching of inside of oct2py,
        #                      # namely changing problem dimension does not
        #                      # correctly redefine calfun_wrapper

        if not re_check:
            sp.io.savemat(filename, Results)


if __name__ == "__main__":
    doit()
