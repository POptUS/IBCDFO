from scipy.io import loadmat


import numpy as np


from calFun import calFun


from pounders import pounders


class Testpounders:
    def test_pounders1(self):
        dataDictionary = loadmat('callpounders.mat')
        dataDictionaryX0 = loadmat('callpoundersX0.mat')
        # Using the default X0 value
        X0 = dataDictionaryX0['X0']
        X = dataDictionary['X']
        F = dataDictionary['F']
        flag = dataDictionary['flag']
        xkin = dataDictionary['xkin']
        func = calFun
        n = 16
        np.random.seed(0)
        mpmax = 2 * n + 1
        nfmax = 200
        gtol = 10 ** -13
        delta = 0.1
        nfs = 0
        m = n
        F0 = []
        xind = 0
        Low = 0.5 * np.ones((1, n))
        Upp = 0.8 * np.ones((1, n))
        printf = False
        # Choose your solver:
        spsolver = 1  # FixMe: Set to global in pounders???
        # spsolver=2
        [XOut, FOut, flagOut, xkinOut] = pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf)
        error = np.linalg.norm(XOut - X, 'fro')
        assert error < 10 ** -20
        print(f'Error in the Frobenius norm between XOut and X in test_pounders1 is {error}\n')
        error = np.linalg.norm(FOut - F, 'fro')
        assert error < 10 ** -20
        print(f'Error in the Frobenius norm between FOut and F in test_pounders1 is {error}\n')
        assert flagOut == flag
        assert xkin == xkinOut
        # [XOut, FOut, flagOut, xkinOut] = pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, -Low, Upp, printf)
        # pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, True)
        # Increase code coverage in first branch statement of pounders
        pounders(float('inf'), X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, True)

    def test_pounders2(self):
        dataDictionary = loadmat('callpounders2.mat')
        # X0 = .5+.1*rand(1,n) in Matlab and rand('state',0)
        X0 = dataDictionary['X0']
        X = dataDictionary['X']
        F = dataDictionary['F']
        flag = dataDictionary['flag']
        xkin = dataDictionary['xkin']
        func = calFun
        n = 16
        np.random.seed(0)
        mpmax = 2 * n + 1
        nfmax = 200
        gtol = 10 ** -13
        delta = 0.1
        nfs = 0
        m = n
        F0 = []
        xind = 0
        Low = 0.5 * np.ones((1, n))
        Upp = 0.8 * np.ones((1, n))
        printf = False
        # Choose your solver:
        spsolver = 1  # FixMe: Set to global in pounders???
        # spsolver=2
        [XOut, FOut, flagOut, xkinOut] = pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf)
        error = np.linalg.norm(XOut - X, 'fro')
        assert error < 10 ** -20
        print(f'Error in the Frobenius norm between XOut and X in test_pounders2 is {error}\n')
        error = np.linalg.norm(FOut - F, 'fro')
        assert error < 10 ** -20
        print(f'Error in the Frobenius norm between FOut and F in test_pounders2 is {error}\n')
        assert flagOut == flag
        assert xkin == xkinOut
        pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, True)

    def test_pounders3(self):
        dataDictionary = loadmat('callpounders3.mat')
        # X0 = .5+.3*rand(1,n) in Matlab and rand('state',0)
        X0 = dataDictionary['X0']
        X = dataDictionary['X']
        F = dataDictionary['F']
        flag = dataDictionary['flag']
        xkin = dataDictionary['xkin']
        func = calFun
        n = 16
        np.random.seed(0)
        mpmax = 2 * n + 1
        nfmax = 200
        gtol = 10 ** -13
        delta = 0.1
        nfs = 0
        m = n
        F0 = []
        xind = 0
        Low = 0.5 * np.ones((1, n))
        Upp = 0.8 * np.ones((1, n))
        printf = False
        # Choose your solver:
        spsolver = 1  # FixMe: Set to global in pounders???
        # spsolver=2
        [XOut, FOut, flagOut, xkinOut] = pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf)
        error = np.linalg.norm(XOut - X, 'fro')
        assert error < 10 ** -20
        print(f'Error in the Frobenius norm between XOut and X in test_pounders3 is {error}\n')
        error = np.linalg.norm(FOut - F, 'fro')
        assert error < 10 ** -20
        print(f'Error in the Frobenius norm between FOut and F in test_pounders3 is {error}\n')
        assert flagOut == flag
        assert xkin == xkinOut
        pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, True)

    def test_pounders4(self):
        dataDictionary = loadmat('callpounders4.mat')
        # X0 = .5+.2*rand(1,n) in Matlab and rand('state',0)
        X0 = dataDictionary['X0']
        X = dataDictionary['X']
        F = dataDictionary['F']
        flag = dataDictionary['flag']
        xkin = dataDictionary['xkin']
        func = calFun
        n = 16
        np.random.seed(0)
        mpmax = 2 * n + 1
        nfmax = 200
        gtol = 10 ** -13
        delta = 0.1
        nfs = 0
        m = n
        F0 = []
        xind = 0
        Low = 0.5 * np.ones((1, n))
        Upp = 0.8 * np.ones((1, n))
        printf = False
        # Choose your solver:
        spsolver = 1  # FixMe: Set to global in pounders???
        # spsolver=2
        [XOut, FOut, flagOut, xkinOut] = pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf)
        error = np.linalg.norm(XOut - X, 'fro')
        assert error < 10 ** -20
        print(f'Error in the Frobenius norm between XOut and X in test_pounders4 is {error}\n')
        error = np.linalg.norm(FOut - F, 'fro')
        assert error < 10 ** -20
        print(f'Error in the Frobenius norm between FOut and F in test_pounders4 is {error}\n')
        assert flagOut == flag
        assert xkin == xkinOut
        pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, True)

    def test_pounders5(self):
        dataDictionary = loadmat('callpounders5.mat')
        # X0 = .5+.25*rand(1,n) in Matlab and rand('state',0)
        X0 = dataDictionary['X0']
        X = dataDictionary['X']
        F = dataDictionary['F']
        flag = dataDictionary['flag']
        xkin = dataDictionary['xkin']
        func = calFun
        n = 16
        np.random.seed(0)
        mpmax = 2 * n + 1
        nfmax = 200
        gtol = 10 ** -13
        delta = 0.1
        nfs = 0
        m = n
        F0 = []
        xind = 0
        Low = 0.5 * np.ones((1, n))
        Upp = 0.8 * np.ones((1, n))
        printf = False
        # Choose your solver:
        spsolver = 1  # FixMe: Set to global in pounders???
        # spsolver=2
        [XOut, FOut, flagOut, xkinOut] = pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf)
        error = np.linalg.norm(XOut - X, 'fro')
        assert error < 10 ** -20
        print(f'Error in the Frobenius norm between XOut and X in test_pounders5 is {error}\n')
        error = np.linalg.norm(FOut - F, 'fro')
        assert error < 10 ** -20
        print(f'Error in the Frobenius norm between FOut and F in test_pounders5 is {error}\n')
        assert flagOut == flag
        assert xkin == xkinOut
        pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, True)

flagTest6 = True  # Run this test
printTest6 = True # Flag to run print statements
# Test case when Low = -0.5 * ones(1, n)
if flagTest6:
    # rand('state', 0) is turned on
    # X0 = 0.5 + 0.1 * rand(1, n)
    # Low = -0.5 * ones(1, n)
    dataDictionary = loadmat('test_callpounders_low.mat')
    matlabDictionary = loadmat('callpounders6.mat')
    X0 = dataDictionary['X0']
    xind = dataDictionary['xind']
    func = calFun
    n = 16
    np.random.seed(0)
    mpmax = 2 * n + 1
    nfmax = 200
    gtol = 10 ** -13
    delta = 0.1
    nfs = 0
    m = n
    F0 = []
    xind = 0
    Low = -0.5 * np.ones((1, n))
    Upp = 0.8 * np.ones((1, n))
    printf = False
    # printf = True
    # Choose your solver:
    spsolver = 1  # FixMe: Set to global in pounders???
    # spsolver=2
    [XOut, FOut, flagOut, xkinOut] = pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf)
    count = 0
    # Tuning parameter to check differences between X, F in both matlab / python
    if printTest6:
        tol = 1 * (10 ** -2)  # Tune this to see how many entries in X, F differ between matlab and python
        maxError = float('-inf')
        for i in range(np.shape(XOut)[0]):
            for j in range(np.shape(XOut)[1]):
                val = XOut[i, j] - matlabDictionary['X'][i, j]
                if val > tol:
                    count += 1
                    maxError = val if val >= maxError else maxError
        matrixError = np.linalg.norm(matlabDictionary['X'] - XOut, 'fro')
        print('\nSimulation #6\n')
        print(f'rand(state, 0)\nX0 = 0.5 + 0.1 *  rand(1, n)\nn is {n}, Low = -0.5 * one(1, n)\n')
        print(f'\nThere are {count} entries in X which is of size {np.shape(XOut)} ({np.shape(XOut)[0] * np.shape(XOut)[1]} entries total)')
        print(f'that differ more than {tol} between Matlab and Python implementation with largest error {maxError} in some entry of X and the error in the frobenius norm is {matrixError}')
        count = 0
        maxError = float('-inf')
        for i in range(np.shape(FOut)[0]):
            for j in range(np.shape(FOut)[1]):
                val = FOut[i, j] - matlabDictionary['F'][i, j]
                if val > tol:
                    count += 1
                    maxError = val if val >= maxError else maxError
        matrixError = np.linalg.norm(matlabDictionary['F'] - FOut, 'fro')
        print(f'\nThere are {count} entries in F which is of size {np.shape(FOut)} ({np.shape(FOut)[0] * np.shape(FOut)[1]} entries total)')
        print(f'that differ more than {tol} between Matlab and Python implementation with largest error {maxError} in some entry of F and the error in the frobenius norm is {matrixError}')

flagTest7 = True  # Run this test
printTest7 = True # Flag to run print statements
# Data generated from matlab using n = 3
# Before pounders function call in callpounders.m, we save data as test_callpounders_low2.mat
# We save output of callpounders.m in callpounders7.mat
if flagTest7:
    dataDictionary = loadmat('test_callpounders_low2.mat')
    matlabDictionary = loadmat('callpounders7.mat')
    X0 = dataDictionary['X0']
    xind = dataDictionary['xind']
    func = calFun
    n = 3
    mpmax = 2 * n + 1
    nfmax = 200
    gtol = 10 ** -13
    delta = 0.1
    nfs = 0
    m = n
    F0 = []
    xind = 0
    Low = -0.5 * np.ones((1, n))
    Upp = 0.8 * np.ones((1, n))
    printf = False
    # printf = True
    # Choose your solver:
    spsolver = 1  # FixMe: Set to global in pounders???
    # spsolver=2
    [XOut, FOut, flagOut, xkinOut] = pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf)
    if printTest7:
        X = matlabDictionary['X']
        F = matlabDictionary['F']
        errorX = np.linalg.norm(X - XOut, 'fro')
        errorF = np.linalg.norm(F - FOut, 'fro')
        print('\nSimulation #7\n')
        print(f'\nSimulation done when n = {3} and when Low = -0.5 * ones(1, {n})\n')
        print(f'The error of X between matlab / Python is {errorX} in the Frobenius norm\n')
        print(f'The error of F between matlab / Python is {errorF} in the Frobenius norm\n')

flagTest8 = True  # Run this test
printTest8 = True # Flag to run print statements
# Data generated from matlab using n = 4
# Before pounders function call in callpounders.m, we save data as test_callpounders_low3.mat
# We save output of callpounders.m in callpounders8.mat
if flagTest8:
    dataDictionary = loadmat('test_callpounders_low3.mat')
    matlabDictionary = loadmat('callpounders8.mat')
    X0 = dataDictionary['X0']
    xind = dataDictionary['xind']
    func = calFun
    n = 4
    mpmax = 2 * n + 1
    nfmax = 200
    gtol = 10 ** -13
    delta = 0.1
    nfs = 0
    m = n
    F0 = []
    xind = 0
    Low = -0.5 * np.ones((1, n))
    Upp = 0.8 * np.ones((1, n))
    printf = False
    # printf = True
    # Choose your solver:
    spsolver = 1  # FixMe: Set to global in pounders???
    # spsolver=2
    [XOut, FOut, flagOut, xkinOut] = pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf)
    if printTest8:
        X = matlabDictionary['X']
        F = matlabDictionary['F']
        errorX = np.linalg.norm(X - XOut, 'fro')
        errorF = np.linalg.norm(F - FOut, 'fro')
        print('\nSimulation #8\n')
        print(f'\nSimulation done when n = {4} and when Low = -0.5 * ones(1, {n})\n')
        print(f'The error of X between matlab / Python is {errorX} in the Frobenius norm\n')
        print(f'The error of F between matlab / Python is {errorF} in the Frobenius norm\n')

flagTest9 = True  # Run this test
printTest9 = True # Flag to run print statements
# Data generated from matlab using n = 4, nfs = 2
# Generate X0 = 0.5 + 0.1 * rand(nfs, n) in callpounders.m and save it in test_callpounders_low4.mat
# Generate F0 by the following steps:
# for nf = 1:nfs
#   F0(nf, :) = calFun(X0(nf, :))
# end
# and saving F0 in test_callpounders_low4.mat
# Before pounders function call in callpounders.m, we save data as test_callpounders_low4.mat
# We save output of callpounders.m in callpounders9.mat
if flagTest9:
    dataDictionary = loadmat('test_callpounders_low4.mat')
    matlabDictionary = loadmat('callpounders9.mat')
    X0 = dataDictionary['X0']
    xind = dataDictionary['xind']
    func = calFun
    n = 4
    mpmax = 2 * n + 1
    nfmax = 200
    gtol = 10 ** -13
    delta = 0.1
    nfs = 2
    m = n
    F0 = dataDictionary['F0']
    xind = 0
    Low = -0.5 * np.ones((1, n))
    Upp = 0.8 * np.ones((1, n))
    printf = False
    # printf = True
    # Choose your solver:
    spsolver = 1  # FixMe: Set to global in pounders???
    # spsolver=2
    [XOut, FOut, flagOut, xkinOut] = pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf)
    if printTest9:
        X = matlabDictionary['X']
        F = matlabDictionary['F']
        errorX = np.linalg.norm(X - XOut, 'fro')
        errorF = np.linalg.norm(F - FOut, 'fro')
        print('\nSimulation #9\n')
        print(f'\nSimulation done when n = {4}, nfs = 2, and when Low = -0.5 * ones(1, {n})\n')
        print(f'\nX0 = 0.5 + 0.1 * rand(nfs, {n})\n\nfor nf = 1:nfs\n\tF0(nf,:) = calFun(X0(nf,:))\nend\n')
        print(f'The error of X between matlab / Python is {errorX} in the Frobenius norm\n')
        print(f'The error of F between matlab / Python is {errorF} in the Frobenius norm\n')

flagTest10 = True  # Run this test
printTest10 = True # Flag to run print statements
# Data generated from matlab using n = 4, nfs = 2
# Generate X0 by the following line in matlab callpounders.m and save it in test_callpounders_low5.mat
# 1. rand('state',0)
# 2. X0 = .5+0.1*rand(1,n);
# Generate X0 & F0 by the following lines in matlab and save them in test_callpounders_low5.mat:
# I = eye(n)
# for nf = 2:nfs
#   X0(nf, :) = X0(1, :) + I(nf-1, :)
# end
# for nf = 1:nfs
#   F0(nf, :) = calFun(X0(nf, :))
# end
# Before pounders function call in callpounders.m, we save data as test_callpounders_low5.mat
# We save output of callpounders.m in callpounders10.mat
if flagTest10:
    dataDictionary = loadmat('test_callpounders_low5.mat')
    matlabDictionary = loadmat('callpounders10.mat')
    X0 = dataDictionary['X0']
    xind = dataDictionary['xind']
    func = calFun
    n = 4
    mpmax = 2 * n + 1
    nfmax = 200
    gtol = 10 ** -13
    delta = 0.1
    nfs = 2
    m = n
    F0 = dataDictionary['F0']
    xind = 0
    Low = -0.5 * np.ones((1, n))
    Upp = 0.8 * np.ones((1, n))
    printf = False
    # printf = True
    # Choose your solver:
    spsolver = 1  # FixMe: Set to global in pounders???
    # spsolver=2
    [XOut, FOut, flagOut, xkinOut] = pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf)
    if printTest10:
        X = matlabDictionary['X']
        F = matlabDictionary['F']
        errorX = np.linalg.norm(X - XOut, 'fro')
        errorF = np.linalg.norm(F - FOut, 'fro')
        print('\nSimulation #10\n')
        print(f'\nSimulation done when n = {4}, nfs = 2, and when Low = -0.5 * ones(1, {n})\n')
        print(f'X0(1, :) predefined by X0 = 0.5 + 0.1 * rand(1, {n}) with rand(state, 0)\n\nLet I = eye(n)\n\nfor nf=2:nfs\n\tX0(nf, :) = X0(1, :) + I(nf-1, :)\nend\n\nfor nf=1:nfs\n\tF0(nf, :) = calFun(X0(nf, :))\nend\n')
        print(f'The error of X between matlab / Python is {errorX} in the Frobenius norm\n')
        print(f'The error of F between matlab / Python is {errorF} in the Frobenius norm\n')
