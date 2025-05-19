import unittest

import ibcdfo.pounders as pdrs
import numpy as np


class TestPhi2EvalUnittest(unittest.TestCase):
    def test_phi2eval(self):
        D = np.eye(3)
        T = np.zeros((3, 6))
        T[0, 0] = 0.5
        T[1, 3] = 0.5
        T[2, 5] = 0.5

        P = pdrs.phi2eval(D)
        self.assertTrue(np.all(P == T))
