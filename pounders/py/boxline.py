import numpy as np


def boxline(D, X, Low, Upp):
    """
    boxline.py                                     Created by Stefan Wild
    Modified 12/7/2009
    This routine finds the smallest t>=0 for which X+t*D hits the box [Low,Upp]
    function boxline(D,X,Low,Upp) -> t
    --INPUTS-----------------------------------------------------------------
    Here are the numpy array
    D      = [dbl] [n-by-1] Direction
    Low      = [dbl] [n-by-1] Lower bounds
    X      = [dbl] [n-by-1] Current Point (assumed to live in [Low,Upp])
    Upp      = [dbl] [n-by-1] Upper bounds
    --OUTPUTS----------------------------------------------------------------
    t      = [dbl] Value of smallest t>=0 for which X+t*D hits a constraint
                   Set to 1 if t=1 does not hit constraint for t<1.
    """
    # Safety for the n=1 case
    Upp = np.atleast_1d(Upp)
    Low = np.atleast_1d(Low)
    # Number of rows in X
    n = np.shape(X)[0]
    t = 1
    for i in range(0, n):
        if D[i] > 0:
            t = min(t, (Upp[i] - X[i]) / D[i])
        elif D[i] < 0:
            t = min(t, (Low[i] - X[i]) / D[i])
    return t
