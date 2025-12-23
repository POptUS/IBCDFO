import ibcdfo
import unittest

import numpy as np

# BenDFO
from calfun import calfun
from dfoxs import dfoxs

from ibcdfo.manifold_sampling.tests.general_smooth_h_funs import h_leastsquares


class TestMSPLeastSquares(unittest.TestCase):
    def testLeastSquares(self):
        # ----- HARDCODED VALUES
        DFO_PROBS = [0, 1]
        NF_MAX = 300
        FACTOR = 10
        SUBPROB_SWITCH = "linprog"

        # ----- SUITE OF PROBLEMS
        DFO = np.loadtxt("dfo.dat")

        # ----- RUN & TEST OPTIMIZATIONS
        for nprob, n, m, factor_power in DFO[DFO_PROBS, :]:
            nprob = int(nprob)
            n = int(n)
            m = int(m)
            LB = np.full(n, -5000.0, float)
            UB = np.full(n, 5000.0, float)
            x0 = dfoxs(n, nprob, FACTOR**factor_power)

            def Ffun(y):
                out = calfun(y, m, nprob, "smooth", 0, num_outs=2)[1]
                assert len(out) == m, "Incorrect output dimension"
                return np.squeeze(out)

            X, F, hF, xkin, flag = ibcdfo.run_MSP(h_leastsquares, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)
            self.assertTrue(hF[xkin] <= 36.0 + 1.0e-8)
