"""
Unit test of compute function
"""

import unittest
from pathlib import Path

import ibcdfo
import numpy as np

_TEST_PATH = Path(__file__).resolve().parent


class TestCompute(unittest.TestCase):
    # def setUp(self):
    #     fname = _TEST_PATH.joinpath("compute_test_data.csv")
    #     self.__tests = np.loadtxt(fname, delimiter=",")

    def testSomething(self):
        self.assertTrue(True)

    # def testAgain(self):
    #     # This is just running all of the tests again...
    #     myt.test()
