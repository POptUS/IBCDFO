"""
Unit test of compute function
"""

import unittest
from pathlib import Path

import ibcdfo.pounders as pdrs
import numpy as np

_TEST_PATH = Path(__file__).resolve().parent


class TestPounders(unittest.TestCase):
    # def setUp(self):
    #     fname = _TEST_PATH.joinpath("compute_test_data.csv")
    #     self.__tests = np.loadtxt(fname, delimiter=",")

    def testSomething(self):
        self.assertTrue(True)

    def test_phi2eval(self):
        D = np.eye(3)
        T = np.zeros((3, 6))
        T[0, 0] = 0.5
        T[1, 3] = 0.5
        T[2, 5] = 0.5

        P = pdrs.phi2eval(D)
        assert np.all(P == T), "Test failed"
