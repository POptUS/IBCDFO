"""
Port of goombah/m/tests/test_minimize_quadratics_lhs.m.
"""

import unittest

import numpy as np
from ibcdfo.goombah import minimize_quadratics_lhs
from ibcdfo.manifold_sampling import h_pw_maximum_squared


class TestMinimizeQuadraticsLhs(unittest.TestCase):
    def test_known_minimizer(self):
        # The linear models z(s) = s - c have a common zero at s = c, which
        # is interior to the bounding box, so the minimum of max(z(s)**2)
        # over the box is 0.
        n = 3
        c = np.array([0.1, -0.2, 0.3])
        Hres = np.zeros((n, n, n))
        Gres = np.eye(n)
        Cres = -c
        Low = -0.5 * np.ones(n)
        Upp = 0.5 * np.ones(n)

        val_at_center, _, _ = h_pw_maximum_squared(Cres)

        rng = np.random.default_rng(1)
        s_k, pred_dec = minimize_quadratics_lhs(Hres, Gres, Cres, Low, Upp, val_at_center, h_pw_maximum_squared, rng=rng)

        self.assertTrue(np.all(Low <= s_k) and np.all(s_k <= Upp), "Step is not within the bounds")
        self.assertLessEqual(abs(pred_dec - val_at_center), 1e-4, "Did not find the known minimum")

    def test_pred_dec_clamped_to_zero(self):
        # The predicted decrease should never be negative, even if x_center
        # is already a global minimizer.
        n = 3
        c = np.array([0.1, -0.2, 0.3])
        Hres = np.zeros((n, n, n))
        Gres = np.eye(n)
        Cres = -c
        Low = -0.5 * np.ones(n)
        Upp = 0.5 * np.ones(n)

        rng = np.random.default_rng(1)
        _, pred_dec = minimize_quadratics_lhs(Hres, Gres, Cres, Low, Upp, -1.0, h_pw_maximum_squared, rng=rng)

        self.assertEqual(pred_dec, 0.0, "Predicted decrease should have been clamped to zero")


if __name__ == "__main__":
    unittest.main()
