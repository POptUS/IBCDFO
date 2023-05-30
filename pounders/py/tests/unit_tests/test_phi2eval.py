#!/usr/bin/env python

import numpy as np
import ibcdfo.pounders as pdrs


class Test_phi2eval:
    def test_phi2eval(self):
        D = np.eye(3)
        T = np.zeros((3, 6))
        T[0, 0] = 0.5
        T[1, 3] = 0.5
        T[2, 5] = 0.5

        P = pdrs.phi2eval(D)
        assert np.all(P == T), "Test failed"


if __name__ == "__main__":
    runner = Test_phi2eval()
    runner.test_phi2eval()
