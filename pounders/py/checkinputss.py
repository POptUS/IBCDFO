import numbers

import numpy as np


def checkinputss(Ffun, X_0, n, np_max, nf_max, g_tol, delta, nfs, m, X_init, F_init, xk_in, Low, Upp):
    """
    THIS DOES NOT ALTER ANY OF THE INPUT ARGUMENTS.

    checkinputss(Ffun,X_0,n,np_max,nf_max,g_tol,delta,nfs,m,F_init,xk_in,Low,Upp) -> [flag,X_0,np_max,F_init,Low,Upp]
    Checks the inputs provided to pounders.
    A warning message is produced if a nonfatal input is given (and the input is changed accordingly).
    An error message (flag=-1) is produced if the pounders cannot continue.
    --INPUTS-----------------------------------------------------------------
    see inputs for pounders.py
    --OUTPUTS----------------------------------------------------------------
    flag  [int] = 1 if inputs pass the test
                = 0 if a warning was produced (X_0,np_max,F_init,Low,Upp are changed)
                = -1 if a fatal error was produced (pounders terminates)
    """
    flag = 1

    # Check Ffun
    if not callable(Ffun):
        print("Error: Ffun is not a function handle")
        return -1
    # Check starting point
    if not isinstance(X_0, np.ndarray):
        print("Error: X_0 must be a Numpy array")
        return -1
    elif (X_0.ndim != 1) or (len(X_0) != n):
        print(f"Error: X_0 is not an {n} element Numpy array")
        return -1
    # Check max number of interpolation points
    if np_max < n + 1 or np_max > int(0.5 * (n + 1) * (n + 2)):
        np_max = max(n + 1, min(np_max, int(0.5 * (n + 1) * (n + 2))))
        print(f"Warning: np_max not in [n+1, 0.5 * (n+1) * (n+2)] using {np_max}")
        flag = 0
    # Check standard positive quantities
    if nf_max < 1:
        print("Error: max number of evaluations is less than 1")
        return -1
    elif (g_tol <= 0.0) or (not np.isfinite(g_tol)):
        print("Error: g_tol must be positive")
        return -1
    elif (delta <= 0.0) or (not np.isfinite(delta)):
        print("Error: delta must be positive")
        return -1
    # Check starting point
    if not isinstance(xk_in, numbers.Integral):
        print("Error: starting point index not an integer")
        return -1
    elif (xk_in < 0) or ((nfs == 0) and (xk_in != 0)) or ((nfs > 0) and (xk_in >= nfs)):
        print("Error: Invalid starting point index")
        return -1
    # Check matrix of initial points and function values
    if (X_init.ndim != 2) or (F_init.ndim != 2):
        print("Error: X_init and F_init must be 2D arrays")
        return -1
    if nfs > 0:
        # Only check sizes and contents if values are provided
        nfs2, n2 = X_init.shape
        if nfs2 != nfs:
            print("Error: number of initial points nfs does not match input X_init")
            return -1
        elif n != n2:
            print("Error: X_init does not contain the right number of coordinates")
            return -1
        if not np.all(np.isfinite(X_init)):
            print("Error: X_init contains non-finite values")
            return -1
        if not np.array_equiv(X_init[xk_in], X_0):
            print("Error: Starting point X_0 doesn't match row in X_init[xk_in]")
            return -1

        nfs2, m2 = F_init.shape
        if nfs2 != nfs:
            print("Error: number of starting f values nfs does not match input F_init")
            return -1
        elif m != m2:
            print("Error: F_init does not contain the right number of residuals")
            return -1
        if not np.all(np.isfinite(F_init)):
            print("Error: F_init contains non-finite values")
            return -1

    # Check the bounds
    if (not isinstance(Low, np.ndarray)) or (not isinstance(Low, np.ndarray)):
        print("Error: Low and Upp must be Numpy arrays")
        return -1
    if (Low.ndim != 1) or (Low.ndim != 1):
        print("Error: Low and Upp must be 1D arrays")
        return -1
    if (len(Low) != n) or len(Low) != len(Upp):
        print(f"Error: Low and Upp are not {n} element 1D arrays")
        return -1
    if np.any(np.isnan(Upp)) or np.any(np.isnan(Low)):
        print("Error: Upp or Low bounds contain non-finite values")
        return -1
    if any(Low >= Upp):
        print("Error: must have Upp > Low")
        return -1
    if any(X_0 < Low) or any(X_0 > Upp):
        print("Error: starting point outside of bounds (Low,Upp)")
        return -1
    return flag
