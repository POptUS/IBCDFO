import numpy as np


def phi2eval(X):
    """
    Input:
    X is a [m-by-n] numpy array
    Output:
    Phi
    """
    [m, n] = np.shape(X)
    Phi = np.zeros((m, int(0.5 * n * (n + 1))))
    X2 = 0.5 * X**2

    j = 0
    for k in range(0, n):
        Phi[:, j] = X2[:, k]
        j += 1
        # Hadamard product of column vectors
        Phi[:, j : j + n - k - 1] = np.multiply(X[:, [k]], X[:, k + 1 :]) / np.sqrt(2)
        j += n - k - 1
    return Phi

def natural_basis(X):

    num_points = np.shape(X)[0]
    Phi = phi2eval(X)

    return np.concatenate((np.ones((num_points, 1)), X, Phi), axis=1)
