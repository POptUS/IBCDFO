"""
Unit test of compute function
"""

import os
import unittest

import ibcdfo.pounders as pdrs
import numpy as np
from calfun import calfun
from dfoxs import dfoxs


class TestPounders(unittest.TestCase):
    def test_benchmark_pounders(self):
        if not os.path.exists("formquad_results"):
            os.makedirs("formquad_results")

        dfo = np.loadtxt("dfo.dat")

        spsolver = 2
        g_tol = 1e-13
        factor = 10
        nf_max = 100

        for row, (nprob, n, m, factor_power) in enumerate(dfo):
            n = int(n)
            m = int(m)

            def Ffun(y):
                out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[1]

                return np.squeeze(out)

            X_0 = dfoxs(n, nprob, int(factor**factor_power))
            Low = -np.inf * np.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
            Upp = np.inf * np.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]
            nfs = 1
            F_init = np.zeros((1, m))
            F_init[0] = Ffun(X_0)
            xind = 0
            delta = 0.1
            printf = False

            Opts = {"printf": printf, "spsolver": spsolver, "row": row}
            Prior = {"nfs": 1, "F_init": F_init, "X_init": X_0, "xk_in": xind}

            X, F, hF, flag, xk_best = pdrs.pounders(Ffun, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Prior=Prior, Options=Opts, Model={})

            evals = F.shape[0]

            self.assertNotEqual(flag, 1, f"pounders failed. (flag={flag})")
            self.assertTrue(X.shape[0] <= nf_max + nfs, f"POUNDERs grew the size of X: X.shape[0]={X.shape[0]}, limit={nf_max + nfs}")

            if flag == 0:
                self.assertTrue(evals <= nf_max + nfs, f"POUNDERs evaluated more than nf_max evaluations: evals={evals}, limit={nf_max + nfs}")
            elif flag != -6 and flag != -4:
                self.assertTrue(evals == nf_max + nfs, f"POUNDERs didn't use nf_max evaluations: evals={evals}, expected={nf_max + nfs}, flag={flag}")
