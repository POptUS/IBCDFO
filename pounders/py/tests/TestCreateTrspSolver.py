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
        N = 1
        G = np.array([-1.1])
        H = np.atleast_2d([2.2])
        Low = np.array([-1.9])
        Upp = np.array([0.9])
        self.assertTrue(H[0, 0] > 0.0)

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
            self.assertEqual(len(s_0), N)
            self.assertTrue(isinstance(f_0, numbers.Real))
            self.assertTrue(np.fabs(1.0 - s_0[0] / s_expected) <= 110.0 * EPS)
            self.assertTrue(np.fabs(1.0 - f_0 / f_expected) <= 110.0 * EPS)

            # Unconstrained solution outside bounds
            s_0, f_0, found_solution = solve_trsp(H, G, Low, too_small)
            self.assertTrue(found_solution)
            self.assertTrue(isinstance(s_0, np.ndarray))
            self.assertEqual(s_0.ndim, 1)
            self.assertEqual(len(s_0), N)
            self.assertTrue(isinstance(f_0, numbers.Real))
            self.assertTrue(np.fabs(1.0 - s_0[0] / s_small) <= 110.0 * EPS)
            self.assertTrue(np.fabs(1.0 - f_0 / f_small) <= 110.0 * EPS)

            if idx != ibcdfo.pounders.constants.TRSP_SOLVER_SIMPLE:
                # The simple sampler requires that Low <= 0 <= Upp
                s_0, f_0, found_solution = solve_trsp(H, G, too_large, Upp)
                self.assertTrue(found_solution)
                self.assertTrue(isinstance(s_0, np.ndarray))
                self.assertEqual(s_0.ndim, 1)
                self.assertEqual(len(s_0), N)
                self.assertTrue(isinstance(f_0, numbers.Real))
                self.assertTrue(np.fabs(1.0 - s_0 / s_large) <= 110.0 * EPS)
                self.assertTrue(np.fabs(1.0 - f_0 / f_large) <= 1500.0 * EPS)

    def test2D(self):
        EPS = np.finfo(float).eps

        # Specify problem
        N = 2
        G = np.array([1.2, -2.3])
        H = np.array([[1.1, -1.2], [-1.2, 4.5]])
        self.assertTrue(np.array_equal(H.T, H, equal_nan=False))
        self.assertTrue(all(np.linalg.eigvalsh(H) > 0.5))
        Low = np.array([-7.0, -4.0])
        Upp = np.array([5.0, 4.0])

        # Known solution
        s_expected = np.array([-0.75213675, 0.31054131])
        f_expected = -0.8084045584045583

        for idx in self.__solvers:
            # Expected emission of warnings tested in testWarnings.  Ignore only those.
            warnings.simplefilter("default")
            for warn_idx, _ in self.__emit_warnings:
                if warn_idx == idx:
                    warnings.simplefilter("ignore")
                    break

            solve_trsp = ibcdfo.pounders.create_trsp_solver(idx)
            self.assertTrue(callable(solve_trsp))

            s_0, f_0, found_solution = solve_trsp(H, G, Low, Upp)
            self.assertTrue(found_solution)
            self.assertTrue(isinstance(s_0, np.ndarray))
            self.assertEqual(s_0.ndim, 1)
            self.assertEqual(len(s_0), N)
            self.assertTrue(isinstance(f_0, numbers.Real))
            max_rel_err = np.max(np.fabs(1.0 - s_0 / s_expected))
            self.assertTrue(max_rel_err <= 5.0e-9)
            rel_err = np.fabs(1.0 - f_0 / f_expected)
            self.assertTrue(rel_err <= 75.0 * EPS)

    def test5D(self):
        # Setting maxit=600,000 in bqmin yielded a solution that was of similar
        # quality to MINQ5's solution.  Since, that's far more that the real
        # budget, we skip it.  We consider a passing 2D test as sufficient
        # evidence of correct functionality for this unofficial solver.
        TO_SKIP = {ibcdfo.pounders.constants.TRSP_SOLVER_SIMPLE}

        # Specify problem
        N = 5
        G = np.array([1.2, -2.3, 0.7, -0.4, 3.4])
        H = np.array([[30.25, 38.5, 115.5, -19.25, 8.25], [38.5, 50.0, 131.0, -14.5, 5.5], [115.5, 131.0, 818.0, -211.5, 67.5], [-19.25, -14.5, -211.5, 388.5, -5.5], [8.25, 5.5, 67.5, -5.5, 64.5]])
        self.assertTrue(np.array_equal(H.T, H, equal_nan=False))
        self.assertTrue(all(np.linalg.eigvalsh(H) > 0.01))
        Low = np.array([-150.0, -10.0, -1.0, -2.0, -1.0])
        Upp = np.array([10.0, 100.0, 3.0, 0.5, 7.0])

        # Known solution
        s_expected = [-128.82782698, 90.82291477, 2.8264531, -1.37446078, 5.60555695]
        f_expected = -170.94945062515922

        for idx in self.__solvers.difference(TO_SKIP):
            solve_trsp = ibcdfo.pounders.create_trsp_solver(idx)
            self.assertTrue(callable(solve_trsp))

            s_0, f_0, found_solution = solve_trsp(H, G, Low, Upp)
            self.assertTrue(found_solution)
            self.assertTrue(isinstance(s_0, np.ndarray))
            self.assertEqual(s_0.ndim, 1)
            self.assertEqual(len(s_0), N)
            self.assertTrue(isinstance(f_0, numbers.Real))
            max_rel_err = np.max(np.fabs(1.0 - s_0 / s_expected))
            self.assertTrue(max_rel_err <= 2.5e-9)
            rel_err = np.fabs(1.0 - f_0 / f_expected)
            self.assertTrue(rel_err <= 7.5e-11)
