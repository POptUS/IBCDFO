from bqmin import bqmin


import numpy as np


from scipy.io import loadmat


class Test_bqmin:
    # Line 226 in pounders.m from callpounders.m
    def test_callpounders(self):
        dictionaryData = loadmat('bqminCallpounders.mat')
        G = dictionaryData['G']
        H = dictionaryData['H']
        Lows = dictionaryData['Lows']
        Upps = dictionaryData['Upps']
        Xsp = dictionaryData['Xsp']
        mdec = dictionaryData['mdec']
        mdec = mdec[0, 0]
        [X, f] = bqmin(H, G, Lows, Upps)
        assert (f - mdec) < 10 ** -10
        assert np.linalg.norm(X - Xsp, float('inf')) < 10 ** -22
