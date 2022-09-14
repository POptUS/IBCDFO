# Test with 'pytest test_calFun'
from calFun import calFun
import numpy as np
from scipy.io import loadmat


class Test_calFun:
    # From call_pounders.m
    def test_calFun1(self):
        dictionaryData = loadmat('calFunCallpounders.mat')
        X0 = dictionaryData['X0']
        Y0 = dictionaryData['Y0']
        assert np.linalg.norm(calFun(X0) - Y0, float('inf')) < 10 ** -6  # Check each entry is within 6 digits

    # From call_pounders_test.py
    def test_calFun2(self):
        X0 = np.array([[0.5, 0.5]])
        Y0 = np.array([[0.75, 0.75]])
        assert np.linalg.norm(calFun(X0) - Y0, float('inf')) < 10 ** -6  # Check each entry within 6 digits
