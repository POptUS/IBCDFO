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


class TestPounders(unittest.TestCase):
    def test_benchmark_pounders(self):
        if not os.path.exists("benchmark_results"):
            os.makedirs("benchmark_results")

        dfo = np.loadtxt("dfo.dat")

        spsolver = 2
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
                out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[1]
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
            if row in [8, 9]:
                printf = True
            else:
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

                filename = "./benchmark_results/pounders4py_nfmax=" + str(nfmax) + "_prob=" + str(row) + "_spsolver=" + str(spsolver) + "_hfun=" + combinemodels.__name__ + ".mat"

                [X, F, flag, xk_best] = pdrs.pounders(objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xind, L, U, printf, spsolver, hfun, combinemodels)

                evals = F.shape[0]
                h = np.zeros(evals)
                for i in range(evals):
                    h[i] = hfun(F[i, :])

                self.assertNotEqual(flag, 1, "pounders failed")
                self.assertTrue(hfun(F[0]) > hfun(F[xk_best]), "No improvement found")
                self.assertTrue(X.shape[0] <= nfmax + nfs, "POUNDERs grew the size of X")

                if flag == 0:
                    self.assertTrue(evals <= nfmax + nfs, "POUNDERs evaluated more than nfmax evaluations")
                elif flag != -4:
                    self.assertTrue(evals == nfmax + nfs, "POUNDERs didn't use nfmax evaluations")

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
