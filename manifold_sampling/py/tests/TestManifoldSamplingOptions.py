"""
Test that manifold_sampling_primal accepts Options and Prior dicts and rejects
ones that are not valid.
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

    def test_prior(self):
        # A second run starts from the points of a first run and only evaluates
        # nf_max new points
        Options = {"printf": 0}
        X1, F1, hF1, _, _ = self.run_msp(self.n + 5, Options)
        i_best = int(np.argmin(hF1))
        Prior = {"nfs": X1.shape[0], "X_init": X1, "F_init": F1, "xk_in": i_best}
        X2, F2, _, _, _ = ibcdfo.run_MSP(self.hfun, self.Ffun, X1[i_best], self.Low, self.Upp, 5, Options, Prior)
        self.assertTrue(np.array_equal(X2[: Prior["nfs"]], X1) and np.array_equal(F2[: Prior["nfs"]], F1))
        self.assertLessEqual(X2.shape[0], Prior["nfs"] + 5)

        # Bad Prior
        with self.assertRaisesRegex(TypeError, "Prior argument must be a dictionary"):
            ibcdfo.run_MSP(self.hfun, self.Ffun, self.x0, self.Low, self.Upp, self.nf_max, Options, "bad")
        with self.assertRaisesRegex(ValueError, "Prior must be a dictionary with the keys"):
            ibcdfo.run_MSP(self.hfun, self.Ffun, X1[i_best], self.Low, self.Upp, self.nf_max, Options, {"nfs": Prior["nfs"]})
        with self.assertRaisesRegex(ValueError, "X_0 doesn't match X_init"):
            ibcdfo.run_MSP(self.hfun, self.Ffun, self.x0 + 1, self.Low, self.Upp, self.nf_max, Options, Prior)
        Prior_repeated = {"nfs": Prior["nfs"] + 1, "X_init": np.vstack((X1, X1[0])), "F_init": np.vstack((F1, F1[0])), "xk_in": i_best}
        with self.assertRaisesRegex(ValueError, "X_init contains repeated points"):
            ibcdfo.run_MSP(self.hfun, self.Ffun, X1[i_best], self.Low, self.Upp, self.nf_max, Options, Prior_repeated)
        Prior_bad_nfs = dict(Prior, nfs=Prior["nfs"] + 1)
        with self.assertRaisesRegex(ValueError, "X_init has shape"):
            ibcdfo.run_MSP(self.hfun, self.Ffun, X1[i_best], self.Low, self.Upp, self.nf_max, Options, Prior_bad_nfs)
