import numpy as np
from ibcdfo.pounders import phi2eval


class Test_phi2eval:
    def test_phi2eval(self):
        D = np.eye(3)
        T = np.zeros((3, 6))
        T[0, 0] = 0.5
        T[1, 3] = 0.5
        T[2, 5] = 0.5

        P = phi2eval(D)
        assert np.all(P == T), "Test failed"


if __name__ == "__main__":
    test_phi2eval()
