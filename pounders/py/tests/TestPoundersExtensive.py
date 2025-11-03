"""
Unit test of compute function
"""

import os
import unittest

import ibcdfo.pounders as pdrs
import ibcdfo.pounders.pounders_concurrent as conc
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
        g_tol = 1e-13
        factor = 10

        for row, (nprob, n, m, factor_power) in enumerate(dfo):
            if row == 0:
                nf_max = 500  # Testing delta_min stopping on first problem
            else:
                nf_max = 50

            n = int(n)
            m = int(m)

            # def Ffun(y):
            #     # It is possible to have python use the same Ffun values as computed by
            #     # matlab via octave.
            #     # Please note this can be slow on some systems.
            #     # This functionality is useful, for example, to test differences between
            #     # matlab and python solvers with a common matlab-computed Ffun.
            #     # To do this, make sure octave is installed on your system and use the
            #     # import statement "from oct2py import octave".
            #     # Then, replace the uncommented line below with
            #     # out = octave.feval("calfun_wrapper", y, m, nprob, "smooth", [], 1, 1)
            #     out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[1]
            #     assert len(out) == m, "Incorrect output dimension"
            #     return np.squeeze(out)

            def Ffun_batch(Y):
                Y = np.atleast_2d(Y)

                out = np.zeros((Y.shape[0], m))  # We will always have a (rows-in-X by 3) output
                for i, y in enumerate(Y):
                    out[i] = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[1]

                return np.squeeze(out)

            X_0 = dfoxs(n, nprob, int(factor**factor_power))
            Low = -np.inf * np.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
            Upp = np.inf * np.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]
            nfs = 1
            F_init = np.zeros((1, m))
            F_init[0] = Ffun_batch(X_0)
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

                X, F, hF, flag, xk_best = pdrs.pounders(Ffun_batch, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Prior=Prior, Options=Opts, Model={})
                Xc, Fc, hFc, flagc, xk_bestc = conc.pounders(Ffun_batch, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Prior=Prior, Options=Opts, Model={})

                self.assertTrue(
                    np.array_equal(X, Xc),
                    f"Mismatch in X between pdrs and conc. "
                    f"Shapes: X={X.shape}, Xc={Xc.shape}. "
                    f"‖X−Xc‖={np.linalg.norm(X - Xc):.3e}. "
                    f"Max diff rows={np.argsort(np.linalg.norm(X - Xc, axis=1))[-3:] if X.ndim>1 else 'N/A'}. "
                    f"Agree rows={np.where(np.linalg.norm(X - Xc, axis=1) == 0)[0] if X.ndim>1 else np.where(X == Xc)[0]}.",
                )

                evals = F.shape[0]

                self.assertNotEqual(flag, 1, "pounders failed")
                self.assertTrue(hfun(F[0]) > hfun(F[xk_best]), "No improvement found")
                self.assertTrue(X.shape[0] <= nf_max + nfs, "POUNDERs grew the size of X")

                if flag == 0:
                    self.assertTrue(evals <= nf_max + nfs, "POUNDERs evaluated more than nf_max evaluations")
                elif flag != -6 and flag != -4:
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
