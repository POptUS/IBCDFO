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
        self.__emit_warnings = {ibcdfo.pounders.constants.TRSP_SOLVER_SIMPLE: ibcdfo.pounders.constants.WARNING_SIMPLE_TRSP}

        warnings.simplefilter("default")

    def testErrors(self):
        for bad in [np.min(list(self.__solvers)) - 1, np.max(list(self.__solvers)) + 1]:
            with self.assertRaises(ValueError):
                ibcdfo.pounders.create_trsp_solver(bad)

    def testWarnings(self):
        for idx in self.__solvers:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ibcdfo.pounders.create_trsp_solver(idx)
                if idx in self.__emit_warnings:
                    self.assertEqual(len(w), 1)
                    self.assertEqual(str(w[0].message), self.__emit_warnings[idx])
                else:
                    self.assertEqual(len(w), 0)

    def test1D(self):
        EPS = np.finfo(float).eps

        # ----- SPECIFY PROBLEMS
        # Unconstrained solution inside bounds
        N = 1
        g = np.array([-1.1])
        H = np.atleast_2d([2.2])
        Low = np.array([-1.9])
        Upp = np.array([0.9])
        self.assertTrue(H[0, 0] > 0.0)

        # Bounds that put unconstrained solution outside bounds
        too_small = np.array([0.25])
        too_large = np.array([0.8])

        # Known solutions
        s_expected = 0.5
        f_expected = -11.0 / 40.0

        s_small = too_small[0]
        f_small = -33.0 / 160.0

        s_large = too_large[0]
        f_large = -22.0 / 125.0

        for idx in self.__solvers:
            # Expected emission of warnings tested in testWarnings.  Ignore only those.
            warnings.simplefilter("default")
            if idx in self.__emit_warnings:
                warnings.filterwarnings("ignore", message=self.__emit_warnings[idx])

            solve_trsp = ibcdfo.pounders.create_trsp_solver(idx)
            self.assertTrue(callable(solve_trsp))

            # Unconstrained solution in bounds
            s_0, f_0, found_solution = solve_trsp(H, g, Low, Upp)
            self.assertTrue(found_solution)
            self.assertTrue(isinstance(s_0, np.ndarray))
            self.assertEqual(s_0.ndim, 1)
            self.assertEqual(len(s_0), N)
            self.assertTrue(isinstance(f_0, numbers.Real))
            self.assertTrue(np.fabs(1.0 - s_0[0] / s_expected) <= 110.0 * EPS)
            self.assertTrue(np.fabs(1.0 - f_0 / f_expected) <= 110.0 * EPS)

            # Unconstrained solution outside bounds
            s_0, f_0, found_solution = solve_trsp(H, g, Low, too_small)
            self.assertTrue(found_solution)
            self.assertTrue(isinstance(s_0, np.ndarray))
            self.assertEqual(s_0.ndim, 1)
            self.assertEqual(len(s_0), N)
            self.assertTrue(isinstance(f_0, numbers.Real))
            self.assertEqual(s_0[0], s_small)
            self.assertTrue(np.fabs(1.0 - f_0 / f_small) <= 75.0 * EPS)

            if idx != ibcdfo.pounders.constants.TRSP_SOLVER_SIMPLE:
                # The simple sampler requires that Low <= 0 <= Upp
                s_0, f_0, found_solution = solve_trsp(H, g, too_large, Upp)
                self.assertTrue(found_solution)
                self.assertTrue(isinstance(s_0, np.ndarray))
                self.assertEqual(s_0.ndim, 1)
                self.assertEqual(len(s_0), N)
                self.assertTrue(isinstance(f_0, numbers.Real))
                self.assertEqual(s_0[0], s_large)
                self.assertTrue(np.fabs(1.0 - f_0 / f_large) <= 1500.0 * EPS)

    def test2D(self):
        # Specify problem
        N = 2
        g = np.array([1.2, -2.3])
        H = np.array([[1.1, -1.2], [-1.2, 4.5]])
        self.assertTrue(np.array_equal(H.T, H, equal_nan=False))
        self.assertTrue(all(np.linalg.eigvalsh(H) > 0.5))
        Low = np.array([-7.0, -4.0])
        Upp = np.array([5.0, 4.0])

        # Known solution
        s_expected = np.array([-88.0 / 117.0, 109.0 / 351.0])
        f_expected = -1135.0 / 1404.0

        for idx in self.__solvers:
            # Expected emission of warnings tested in testWarnings.  Ignore only those.
            warnings.simplefilter("default")
            if idx in self.__emit_warnings:
                warnings.filterwarnings("ignore", message=self.__emit_warnings[idx])

            solve_trsp = ibcdfo.pounders.create_trsp_solver(idx)
            self.assertTrue(callable(solve_trsp))

            s_0, f_0, found_solution = solve_trsp(H, g, Low, Upp)
            self.assertTrue(found_solution)
            self.assertTrue(isinstance(s_0, np.ndarray))
            self.assertEqual(s_0.ndim, 1)
            self.assertEqual(len(s_0), N)
            self.assertTrue(isinstance(f_0, numbers.Real))
            max_rel_err = np.max(np.fabs(1.0 - s_0 / s_expected))
            # print(max_rel_err)
            self.assertTrue(max_rel_err <= 2.5e-13)
            rel_err = np.fabs(1.0 - f_0 / f_expected)
            # print(rel_err)
            self.assertTrue(rel_err <= 2.5e-14)

    def test5D(self):
        # Setting maxit=600,000 in bqmin yielded a solution that was of similar
        # quality to MINQ5's solution.  Since, that's far more that the real
        # budget, we skip it.  We consider a passing 2D test as sufficient
        # evidence of correct functionality for that unofficial solver.
        TO_SKIP = {ibcdfo.pounders.constants.TRSP_SOLVER_SIMPLE}

        # Specify problem
        N = 5
        g = np.array([1.2, -2.3, 0.7, -0.4, 3.4])
        H = np.array([[30.25, 38.5, 115.5, -19.25, 8.25], [38.5, 50.0, 131.0, -14.5, 5.5], [115.5, 131.0, 818.0, -211.5, 67.5], [-19.25, -14.5, -211.5, 388.5, -5.5], [8.25, 5.5, 67.5, -5.5, 64.5]])
        self.assertTrue(np.array_equal(H.T, H, equal_nan=False))
        self.assertTrue(all(np.linalg.eigvalsh(H) > 0.01))
        Low = np.array([-150.0, -10.0, -1.0, -2.0, -1.0])
        Upp = np.array([10.0, 100.0, 3.0, 0.5, 7.0])

        # Known solution
        s_expected = [-18486334673.0 / 143496441.0, 1184796821.0 / 13045131.0, 368714509.0 / 130451310.0, -16300019.0 / 11859210.0, 2014469.0 / 359370.0]
        f_expected = -4906127551123.0 / 28699288200.0

        for idx in self.__solvers.difference(TO_SKIP):
            # Expected emission of warnings tested in testWarnings.  Ignore only those.
            warnings.simplefilter("default")
            if idx in self.__emit_warnings:
                warnings.filterwarnings("ignore", message=self.__emit_warnings[idx])

            solve_trsp = ibcdfo.pounders.create_trsp_solver(idx)
            self.assertTrue(callable(solve_trsp))

            s_0, f_0, found_solution = solve_trsp(H, g, Low, Upp)
            self.assertTrue(found_solution)
            self.assertTrue(isinstance(s_0, np.ndarray))
            self.assertEqual(s_0.ndim, 1)
            self.assertEqual(len(s_0), N)
            self.assertTrue(isinstance(f_0, numbers.Real))
            max_rel_err = np.max(np.fabs(1.0 - s_0 / s_expected))
            # print(max_rel_err)
            self.assertTrue(max_rel_err <= 7.5e-11)
            rel_err = np.fabs(1.0 - f_0 / f_expected)
            # print(rel_err)
            self.assertTrue(rel_err <= 7.5e-11)
