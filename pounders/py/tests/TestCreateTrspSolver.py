"""
Unit test of create_trsp_solver()
"""

import warnings
import unittest

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

    def testSuccessful(self):
        for idx in self.__solvers:
            # Expected emission of warnings tested in testWarnings.  Ignore only those.
            warnings.simplefilter("default")
            for warn_idx, _ in self.__emit_warnings:
                if warn_idx == idx:
                    warnings.simplefilter("ignore")
                    break

            solve_trsp = ibcdfo.pounders.create_trsp_solver(idx)
            self.assertTrue(callable(solve_trsp))
