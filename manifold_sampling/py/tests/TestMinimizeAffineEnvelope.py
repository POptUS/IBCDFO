"""
Check that minimize_affine_envelope does not modify the caller's Lows and Upps.
"""

import unittest

import numpy as np
from ibcdfo.manifold_sampling.minimize_affine_envelope import minimize_affine_envelope


class TestMinimizeAffineEnvelope(unittest.TestCase):
    def test_bounds_are_not_modified(self):
        # One affine piece with gradient e_1, so only the lower bound on s_1 is
        # active; the bounds on s_2 have zero duals.
        n = 2
        G_k = np.array([[1.0], [0.0]])
        Lows = -0.5 * np.ones(n)
        Upps = 0.5 * np.ones(n)
        Lows_in = Lows.copy()
        Upps_in = Upps.copy()

        minimize_affine_envelope(0.0, np.zeros(1), np.zeros(1), G_k, np.zeros((n, n)), 0.5, Lows, Upps, np.zeros((1, n + 1, n + 1)), "linprog", [])

        self.assertTrue(np.array_equal(Lows, Lows_in), "Lows was modified")
        self.assertTrue(np.array_equal(Upps, Upps_in), "Upps was modified")


if __name__ == "__main__":
    unittest.main()
