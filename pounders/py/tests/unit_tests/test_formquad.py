import numpy as np


from formquad import formquad


from scipy.io import loadmat


class Test_formquad:
    # Line 136 Pounders.m from callpounders.m using default X0 in callpounders.m
    def test_formquad1(self):
        dictionaryData = loadmat('formquadCallPounders.mat')
        X = dictionaryData['X']
        F = dictionaryData['F']
        delta = dictionaryData['delta']
        xkin = dictionaryData['xkin']
        mpmax = dictionaryData['mpmax']
        Pars = dictionaryData['Pars']
        vf = dictionaryData['vf']
        Mdir = dictionaryData['Mdir']
        mp = dictionaryData['mp']
        mp = mp[0, 0]
        valid = dictionaryData['valid']
        valid = True if valid else False  # if valid = 1 in matlab, set it to True
        G = dictionaryData['G']
        H = dictionaryData['H']
        Mind = dictionaryData['Mind']
        [MdirOut, mpOut, validOut, GOut, HOut, MindOut] = formquad(X, F, delta, xkin, mpmax, Pars, vf)
        assert mpOut == mp
        assert np.shape(GOut) == np.shape(G)
        assert np.shape(HOut) == np.shape(H)
        assert MindOut == Mind
        assert validOut == valid
        assert np.linalg.norm(MdirOut - Mdir, 'fro') < 10 ** -10

    # Line 154 Pounders.m from callpounders.m using default X0 in callpounders.m
    def test_formquad2(self):
        dictionaryData = loadmat('formquadCallPounders2.mat')
        X = dictionaryData['X']
        F = dictionaryData['F']
        delta = dictionaryData['delta']
        xkin = dictionaryData['xkin']
        mpmax = dictionaryData['mpmax']
        Pars = dictionaryData['Pars']
        vf = dictionaryData['vf']
        mp = dictionaryData['mp']
        mp = mp[0, 0]
        valid = dictionaryData['valid']
        valid = True if valid else False  # if valid = 1 in matlab, set it to True
        Gres = dictionaryData['Gres']
        Hresdel = dictionaryData['Hresdel']
        Mind = dictionaryData['Mind']
        [_, mpOut, validOut, GOut, HOut, MindOut] = formquad(X, F, delta, xkin, mpmax, Pars, vf)
        assert mpOut == mp
        assert np.linalg.norm(Gres - GOut) < 10 ** -10
        assert np.linalg.norm(Hresdel - HOut) < 10 ** -10
        assert sum(MindOut - Mind) == 0  # Check indices are the same
        assert validOut == valid

    # Line 192 Pounders.m from callpounders.m using default X0 in callpounders.m
    def test_formquad3(self):
        dictionaryData = loadmat('formquadCallPounders3.mat')
        X = dictionaryData['X']
        F = dictionaryData['F']
        delta = dictionaryData['delta']
        xkin = dictionaryData['xkin']
        mpmax = dictionaryData['mpmax']
        Pars = dictionaryData['Pars']
        vf = dictionaryData['vf']
        mp = dictionaryData['mp']
        mp = mp[0, 0]
        valid = dictionaryData['valid']
        valid = True if valid else False  # if valid = 1 in matlab, set it to True
        G = dictionaryData['G']
        H = dictionaryData['H']
        Mind = dictionaryData['Mind']
        Mdir = dictionaryData['Mdir']
        [MdirOut, mpOut, validOut, GOut, HOut, MindOut] = formquad(X, F, delta, xkin, mpmax, Pars, vf)
        assert mpOut == mp
        assert np.linalg.norm(G - GOut) < 10 ** -10
        assert np.linalg.norm(H - HOut) < 10 ** -10
        assert sum(abs(MindOut - Mind)) == 0
        assert validOut == valid
        assert np.linalg.norm(Mdir - MdirOut, 'fro') < 10 ** -10

    # Line 205 Pounders.m from callpounders.m using default X0 in callpounders.m
    def test_formquad4(self):
        dictionaryData = loadmat('formquadCallPounders4.mat')
        X = dictionaryData['X']
        F = dictionaryData['F']
        delta = dictionaryData['delta']
        xkin = dictionaryData['xkin']
        mpmax = dictionaryData['mpmax']
        Pars = dictionaryData['Pars']
        vf = dictionaryData['vf']
        mp = dictionaryData['mp']
        mp = mp[0, 0]
        valid = dictionaryData['valid']
        valid = True if valid else False  # if valid = 1 in matlab, set it to True
        G = dictionaryData['G']
        H = dictionaryData['H']
        Mind = dictionaryData['Mind']
        Mdir = dictionaryData['Mdir']
        [MdirOut, mpOut, validOut, GOut, HOut, MindOut] = formquad(X, F, delta, xkin, mpmax, Pars, vf)
        assert mpOut == mp - 1
        assert np.linalg.norm(G - GOut) < 10 ** -10
        assert np.linalg.norm(H - HOut) < 10 ** -10
        assert sum(abs(MindOut - (Mind-1))) == 0
        assert validOut == valid
        assert np.linalg.norm(Mdir - MdirOut, 'fro') < 10 ** -10
