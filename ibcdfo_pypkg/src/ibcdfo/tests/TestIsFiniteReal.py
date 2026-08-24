import unittest

import numpy as np

from ibcdfo._variable_checks import is_finite_real


class TestIsFiniteReal(unittest.TestCase):

    def testFunction(self):
        GOOD_REALS = {
            np.float64(0.0),
            np.finfo(float).smallest_normal,
            np.finfo(float).eps,
            np.float64(1.0),
            1.0 + np.finfo(float).eps,
            np.finfo(float).max,
        }
        NOT_FINITE_REAL = (
            None,
            "",
            "not_finite_real",
            True,
            False,
            1j,
            (1.1 + 1j),
            (0.0 * 1j),
            (1.1 + 0.0 * 1j),
            np.nan,
            np.inf,
            -np.inf,
            {},
            [1.1],
            {1.1},
            np.array(1.1),
            np.array([1.1]),
        )

        for good in GOOD_REALS:
            self.assertTrue(isinstance(good, np.float64))
            self.assertTrue(is_finite_real(good))
            self.assertTrue(is_finite_real(float(good)))
            self.assertTrue(is_finite_real(-good))
            self.assertTrue(is_finite_real(float(-good)))

        for good in np.linspace(-1234.567, 9876.543, 10_000):
            self.assertTrue(isinstance(good, np.float64))
            self.assertTrue(is_finite_real(good))
            self.assertTrue(is_finite_real(float(good)))
            self.assertTrue(is_finite_real(np.float32(good)))

        for bad in NOT_FINITE_REAL:
            self.assertFalse(is_finite_real(bad))
