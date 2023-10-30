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
        nf_max = 50
        g_tol = 1e-13
        factor = 10

        for row, (nprob, n, m, factor_power) in enumerate(dfo):
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

                filename = "./benchmark_results/pounders4py_nf_max=" + str(nf_max) + "_prob=" + str(row) + "_spsolver=" + str(spsolver) + "_hfun=" + combinemodels.__name__ + ".mat"
                Opts = {"printf": printf, "spsolver": spsolver, "hfun": hfun, "combinemodels": combinemodels}
                Prior = {"nfs": 1, "F_init": F_init, "X_init": X_0, "xk_in": xind}

                [X, F, flag, xk_best] = pdrs.pounders(Ffun, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Prior=Prior, Options=Opts, Model={})

                evals = F.shape[0]
                h = np.zeros(evals)
                for i in range(evals):
                    h[i] = hfun(F[i, :])

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
                Results["pounders4py_" + str(row) + "_" + str(hfun_cases)]["H"] = h
                Results["pounders4py_" + str(row) + "_" + str(hfun_cases)]["X"] = X
                # oct2py.kill_octave() # This is necessary to restart the octave instance,
                #                      # and thereby remove some caching of inside of oct2py,
                #                      # namely changing problem dimension does not
                #                      # correctly redefine calfun_wrapper

                sp.io.savemat(filename, Results)
