"""
Test that manifold_sampling_primal accepts an Options dict and rejects one
that is not valid.
"""

import contextlib
import io
import unittest

import ibcdfo
import numpy as np


class TestManifoldSamplingOptions(unittest.TestCase):
    def setUp(self):
        self.n = 3
        self.x0 = np.array([10.0, 20.0, 30.0])
        self.Low = -np.inf * np.ones(self.n)
        self.Upp = np.inf * np.ones(self.n)
        self.hfun = ibcdfo.manifold_sampling.h_pw_maximum_squared
        self.Ffun = lambda x: np.copy(x)
        self.nf_max = 30

    def run_msp(self, nf_max, Options):
        return ibcdfo.run_MSP(self.hfun, self.Ffun, self.x0, self.Low, self.Upp, nf_max, Options)

    def test_bad_options(self):
        with self.assertRaisesRegex(TypeError, "Options argument must be a dictionary"):
            self.run_msp(self.nf_max, "bad")
        with self.assertRaisesRegex(ValueError, "Options dictionary contains unknown keys"):
            self.run_msp(self.nf_max, {"not_an_option": 1})
        with self.assertRaisesRegex(ValueError, "Options subprob_switch must be 'linprog'"):
            self.run_msp(self.nf_max, {"subprob_switch": "GAMS_LP"})

    def test_default_options(self):
        # Omitting Options and passing None or {} are equivalent
        with contextlib.redirect_stdout(io.StringIO()):
            X1 = ibcdfo.run_MSP(self.hfun, self.Ffun, self.x0, self.Low, self.Upp, self.nf_max)[0]
            X2 = self.run_msp(self.nf_max, None)[0]
            X3 = self.run_msp(self.nf_max, {})[0]
            X4 = self.run_msp(self.nf_max, {"subprob_switch": "linprog"})[0]
        self.assertTrue(np.array_equal(X1, X2) and np.array_equal(X1, X3) and np.array_equal(X1, X4))

    def test_printf(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.run_msp(self.nf_max, {"printf": 0})
        self.assertEqual(out.getvalue(), "")

    def test_delta_0(self):
        X = self.run_msp(self.nf_max, {"printf": 0, "delta_0": 0.5})[0]
        self.assertLessEqual(np.max(np.abs(X[1] - self.x0)), 0.5 + 1e-12)

    def test_flags(self):
        # Exhausting the budget gives flag 1
        flag = self.run_msp(self.n + 2, {"printf": 0})[4]
        self.assertEqual(flag, 1)

        # Setting delta_min = delta_0 means that the main loop is never
        # entered, so chi_k is never computed.
        Options = {"printf": 0, "delta_0": 0.1, "delta_min": 0.1}
        flag = self.run_msp(self.nf_max, Options)[4]
        self.assertEqual(flag, -6)

        # checkinputss requires a finite g_tol, so a large one is satisfied by
        # any stationary measure. Starting near the minimizer, the first step
        # fails and delta shrinks below delta_min, which is between gamma_dec *
        # delta_0 and delta_0, so MSP stops with flag 0.
        x0_near = np.array([0.03, 0.04, 0.05])
        Options = {"printf": 0, "delta_0": 0.1, "delta_min": 0.09, "g_tol": 1e10}
        flag = ibcdfo.run_MSP(self.hfun, self.Ffun, x0_near, self.Low, self.Upp, self.nf_max, Options)[4]
        self.assertEqual(flag, 0)
