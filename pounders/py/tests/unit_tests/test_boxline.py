from boxline import boxline


import numpy as np


from scipy.io import loadmat


class Test_boxline:
    # Line 139 pounders.m from callpounders.m
    def test_callPounders1(self):
        dictionaryData = loadmat('boxlineCallpounders.mat')
        X = dictionaryData['X']
        Modeld = dictionaryData['Modeld']
        Low = dictionaryData['Low']
        Upp = dictionaryData['Upp']
        delta = dictionaryData['delta']
        # theta = dictionaryData['theta']
        T = dictionaryData['T']
        for i in range(np.shape(T)[1]):
            assert T[0, i] == boxline(delta * Modeld[i, :], X, Low, Upp)
            assert T[1, i] == boxline(-delta * Modeld[i, :], X, Low, Upp)
    # Line 194 pounders.m from callpounders.m
    def test_callPounders2(self):
        dictionaryData = loadmat('boxlineCallpounders2.mat')
        X = dictionaryData['X']
        Modeld = dictionaryData['Modeld']
        Low = dictionaryData['Low']
        Upp = dictionaryData['Upp']
        delta = dictionaryData['delta']
        # theta = dictionaryData['theta']
        T = dictionaryData['T']
        for i in range(np.shape(T)[1]):
            assert T[0, i] == boxline(delta * Modeld[i, :], X, Low, Upp)
            assert T[1, i] == boxline(-delta * Modeld[i, :], X, Low, Upp)
    # Line 294 & 302 - Not covered
