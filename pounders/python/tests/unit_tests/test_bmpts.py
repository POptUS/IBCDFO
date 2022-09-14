from bmpts import bmpts


import numpy as np


from scipy.io import loadmat


class Test_bmpts:
    # Line 139 pounders.m from callpounders.m
    def test_callPounders1(self):
        dictionaryData = loadmat('bmptsCallpounders.mat')
        X = dictionaryData['X']
        Modeld = dictionaryData['Modeld']
        Low = dictionaryData['Low']
        Upp = dictionaryData['Upp']
        delta = dictionaryData['delta'][0, 0]
        mp = dictionaryData['mp'][0, 0]
        theta = dictionaryData['theta'][0, 0]
        MdirOutput = dictionaryData['MdirOutput']
        [output1, output2] = bmpts(X, Modeld, Low, Upp, delta, theta)
        assert output2 == mp
        assert np.linalg.norm(output1 - MdirOutput, 'fro') < 10 ** -20

    # Line 194 pounders.m from callpounders.m
    def test_callPounders2(self):
        dictionaryData = loadmat('bmptsCallpounders2.mat')
        X = dictionaryData['X']
        Modeld = dictionaryData['Modeld']
        Low = dictionaryData['Low']
        Upp = dictionaryData['Upp']
        delta = dictionaryData['delta'][0, 0]
        mp = dictionaryData['mp'][0, 0]
        theta = dictionaryData['theta'][0, 0]
        MdirOutput = dictionaryData['MdirOutput']
        [output1, output2] = bmpts(X, Modeld, Low, Upp, delta, theta)
        assert output2 == mp
        assert np.linalg.norm(output1 - MdirOutput, 'fro') < 10 ** -20
