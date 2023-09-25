"""
Unit test of compute function
"""

import unittest

import ibcdfo.manifold_sampling as msp
from ibcdfo.manifold_sampling.h_examples.pw_maximum import pw_maximum
import numpy as np


class TestManifoldSampling(unittest.TestCase):
    def test_failing_objective(self):
        def failing_objective(x):
            fvec = x

            if np.random.uniform() < 0.1:
                fvec[0] = np.nan

            return fvec

        subprob_switch = "linprog"
        nfmax = 1000
        X0 = np.array([10, 20, 30])
        L = -np.inf * np.ones(3)
        U = np.inf * np.ones(3)

        np.random.seed(1)

        try:
            [X, F, h, xk_best, flag] = msp.manifold_sampling_primal(pw_maximum, failing_objective, X0, L, U, nfmax, subprob_switch)
        except ValueError:
            assert 1, "Failed like it should have"
        else:
            assert 0, "Didn't fail like it should have"

        L = np.append(L, L)
        [X, F, h, xk_best, flag] = msp.manifold_sampling_primal(pw_maximum, failing_objective, X0, L, U, nfmax, subprob_switch)
        self.assertEqual(flag, -1, "We are testing proper failure of pounders")
