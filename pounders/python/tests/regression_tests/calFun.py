import numpy as np


def calFun(x):
    '''
    Input:
        x is a numpy array (column / row vector)
    Output:
        x + x^2 as a row vector
    '''
    if np.shape(x)[0] > 1:
        x = np.reshape(x, (1, max(np.shape(x))))
    return x + (x ** 2)
