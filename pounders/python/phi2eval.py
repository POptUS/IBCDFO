import numpy as np


def phi2eval(X):
    '''
    Input:
    X is a [m-by-n] numpy array
    Output:
    Phi
    '''
    [m, n] = np.shape(X)
    Phi = np.zeros((m, int(0.5 * n * (n + 1))))
    X2 = 0.5 * X**2

    j = 0
    for k in range(0, n):
        Phi[:, j] = X2[:, k]
        j += 1
        # Hadamard product of column vectors
        Phi[:, j:j + n - k - 1] = np.multiply(np.tile(X[:, k],(n-k-1,1)).T, X[:, k + 1:]) / np.sqrt(2)
        j += n - k - 1
    return Phi
