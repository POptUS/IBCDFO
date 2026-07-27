"""
Unit test of create_trsp_solver()
"""

import numbers
import warnings
import unittest

import numpy as np

import ibcdfo


class TestCreateTrspSolver(unittest.TestCase):
    def setUp(self):
        self.__solvers = {ibcdfo.pounders.constants.TRSP_SOLVER_SIMPLE, ibcdfo.pounders.TRSP_SOLVER_MINQ5}
        self.__emit_warnings = {(ibcdfo.pounders.constants.TRSP_SOLVER_SIMPLE, ("testing", "debugging"))}

    def testErrors(self):
        for bad in [0, ibcdfo.pounders.TRSP_SOLVER_MINQ8]:
            with self.assertRaises(ValueError):
                ibcdfo.pounders.create_trsp_solver(bad)

    def testWarnings(self):
        for idx, words in self.__emit_warnings:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ibcdfo.pounders.create_trsp_solver(idx)
                self.assertEqual(len(w), 1)
                for each in words:
                    self.assertTrue(each in str(w[0].message))

    def test1D(self):
        EPS = np.finfo(float).eps

        # Specify problem
        G = np.array([-1.1])
        H = np.atleast_2d([2.2])
        Low = np.array([-1.9])
        Upp = np.array([0.9])
        self.assertTrue(H[0] > 0.0)

        # Known solutions
        s_expected = 0.5
        f_expected = -0.275

        too_small = np.array([0.25])
        s_small = too_small[0]
        f_small = -0.20625

        too_large = np.array([0.8])
        s_large = too_large[0]
        f_large = -0.176

        for idx in self.__solvers:
            # Expected emission of warnings tested in testWarnings.  Ignore only those.
            warnings.simplefilter("default")
            for warn_idx, _ in self.__emit_warnings:
                if warn_idx == idx:
                    warnings.simplefilter("ignore")
                    break

            solve_trsp = ibcdfo.pounders.create_trsp_solver(idx)
            self.assertTrue(callable(solve_trsp))

            # Unconstrained solution in bounds
            s_0, f_0, found_solution = solve_trsp(H, G, Low, Upp)
            self.assertTrue(found_solution)
            self.assertTrue(isinstance(s_0, np.ndarray))
            self.assertEqual(s_0.ndim, 1)
            self.assertEqual(len(s_0), 1)
            self.assertTrue(isinstance(f_0, numbers.Real))
            self.assertTrue(np.fabs(1.0 - s_0[0] / s_expected) <= 110.0 * EPS)
            self.assertTrue(np.fabs(1.0 - f_0 / f_expected) <= 110.0 * EPS)

            # Unconstrained solution outside bounds
            s_0, f_0, found_solution = solve_trsp(H, G, Low, too_small)
            self.assertTrue(found_solution)
            self.assertTrue(isinstance(s_0, np.ndarray))
            self.assertEqual(s_0.ndim, 1)
            self.assertEqual(len(s_0), 1)
            self.assertTrue(isinstance(f_0, numbers.Real))
            self.assertTrue(np.fabs(1.0 - s_0[0] / s_small) <= 110.0 * EPS)
            self.assertTrue(np.fabs(1.0 - f_0 / f_small) <= 110.0 * EPS)

            if idx != ibcdfo.pounders.constants.TRSP_SOLVER_SIMPLE:
                # The simple sampler requires that Low <= 0 <= Upp
                s_0, f_0, found_solution = solve_trsp(H, G, too_large, Upp)
                self.assertTrue(found_solution)
                self.assertTrue(isinstance(s_0, np.ndarray))
                self.assertEqual(s_0.ndim, 1)
                self.assertTrue(isinstance(f_0, numbers.Real))
                self.assertTrue(np.fabs(1.0 - s_0 / s_large) <= 110.0 * EPS)
                self.assertTrue(np.fabs(1.0 - f_0 / f_large) <= 1500.0 * EPS)
