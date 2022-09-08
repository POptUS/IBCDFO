import numpy as np


def boxline(D, X, L, U):
    '''
    boxline.py                                     Created by Stefan Wild
    Modified 12/7/2009
    This routine finds the smallest t>=0 for which X+t*D hits the box [L,U]
    function boxline(D,X,L,U) -> t
    --INPUTS-----------------------------------------------------------------
    Here are the numpy array
    D      = [dbl] [n-by-1] Direction
    L      = [dbl] [n-by-1] Lower bounds
    X      = [dbl] [n-by-1] Current Point (assumed to live in [L,U])
    U      = [dbl] [n-by-1] Upper bounds
    --OUTPUTS----------------------------------------------------------------
    t      = [dbl] Value of smallest t>=0 for which X+t*D hits a constraint
                   Set to 1 if t=1 does not hit constraint for t<1.
    '''
    # Number of rows in X
    n = np.shape(X)[0]
    t = 1
    for i in range(0, n):
        if D[i] > 0:
            t = min(t, (U[i] - X[i]) / D[i])
        elif D[i] < 0:
            t = min(t, (L[i] - X[i]) / D[i])
    return t
