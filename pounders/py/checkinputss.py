import numpy as np


def checkinputss(Ffun, X_0, n, np_max, nf_max, g_tol, delta, nfs, m, X_init, F_init, xk_in, Low, Upp):
    """
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

    flag = 1  # By default, everything is OK
    if not callable(Ffun):
        print("Error: Ffun is not a function handle")
        flag = -1
        return [flag, X_0, np_max, F_init, Low, Upp, xk_in]
    # Verify X_0 is the appropriate size
    X_0 = np.atleast_2d(X_0)
    assert X_0.shape == (1, n) or X_0.shape == (n, 1), "X_0 is not the right shape"
    F_init = np.atleast_2d(F_init)
    Low = np.atleast_2d(Low)
    Upp = np.atleast_2d(Upp)
    xk_in = int(xk_in)
    [nfs2, n2] = np.shape(X_0)
    if n != n2:
        # Attempt to transpose:
        if n2 == 1 and nfs2 == n:
            X_0 = X_0.T
            print("Warning: X_0 is n-by-1 column vector, using row vector X_0")
            flag = 0
        else:
            print("Error: np.shape(X_0)[1] != n")
            flag = -1
            return [flag, X_0, np_max, F_init, Low, Upp, xk_in]

    if len(X_init):
        assert (X_init.shape[0] == nfs) and (F_init.shape[0] == nfs), "Prior X_init and F_init must have nfs rows"
        assert np.array_equiv(np.atleast_2d(X_init)[xk_in], X_0), "Starting point X_0 doesn't match row in Prior['X_init']"

    # Check max number of interpolation points
    if np_max < n + 1 or np_max > int(0.5 * (n + 1) * (n + 2)):
        np_max = max(n + 1, min(np_max, int(0.5 * (n + 1) * (n + 2))))
        print(f"Warning: np_max not in [n+1, 0.5 * (n+1) * (n+2) using {np_max}")
        flag = 0
    # Check standard positive quantities
    if nf_max < 1:
        print("Error: max number of evaluations is less than 1")
        flag = -1
        return [flag, X_0, np_max, F_init, Low, Upp, xk_in]
    elif g_tol <= 0:
        print("Error: g_tol must be positive")
        flag = -1
        return [flag, X_0, np_max, F_init, Low, Upp, xk_in]
    elif delta <= 0:
        print("Error: delta must be positive")
        flag = -1
        return [flag, X_0, np_max, F_init, Low, Upp, xk_in]
    # Check matrix of initial function values
    # Only check sizes if values are provided
    if nfs > 0:
        [nfs2, m2] = np.shape(F_init)
        if nfs2 < nfs:
            print("Error: fewer than nfs function values in F_init")
            flag = -1
            return [flag, X_0, np_max, F_init, Low, Upp, xk_in]
        elif nfs > 1 and m != m2:
            print("Error: F_init does not contain the right number of residuals")
            flag = -1
            return [flag, X_0, np_max, F_init, Low, Upp, xk_in]
        elif nfs2 > nfs:
            print("Warning: number of starting f values nfs does not match input F_init")
            flag = 0
        if np.any(np.isnan(F_init)):
            print("Error: F_init contains a NaN")
            flag = -1
            return [flag, X_0, np_max, F_init, Low, Upp, xk_in]

    # Check starting point
    if (xk_in > max(nfs - 1, 0)) or (xk_in < 0) or (xk_in % 1 != 0):  # FixMe: Check what xk_in needs to be...
        print("Error: starting point index not an integer between 0 and nfs-1")
        flag = -1
        return [flag, X_0, np_max, F_init, Low, Upp, xk_in]
    # Check the bounds
    [nfs2, n2] = np.shape(Low)
    [nfs3, n3] = np.shape(Upp)
    if (n3 != n2) or (nfs2 != nfs3):
        print("Error: bound dimensions inconsistent")
        flag = -1
        return [flag, X_0, np_max, F_init, Low, Upp, xk_in]
    elif n2 != n and (n2 == 1 and nfs2 == n):
        Low = Low.T
        Upp = Upp.T
        print("Warning: bounds are n-by-1, using transposed row vectors")
        flag = 0
    elif n2 != n or nfs2 != 1:
        print("Error: bounds are not 1-by-n vectors")
        flag = -1
        return [flag, X_0, np_max, F_init, Low, Upp, xk_in]
    if np.any(np.isnan(Upp)) or np.any(np.isnan(Low)):
        print("Error: Upp or Low bounds contain a NaN")
        flag = -1
        return [flag, X_0, np_max, F_init, Low, Upp, xk_in]
    if np.min(Upp - Low) <= 0:
        print("Error: must have Upp > Low")
        flag = -1
        return [flag, X_0, np_max, F_init, Low, Upp, xk_in]
    if np.min([np.min(X_0[xk_in, :] - Low), np.min(Upp - X_0[xk_in, :])]) < 0:
        print("Error: starting point outside of bounds (Low,Upp)")
        flag = -1
        return [flag, X_0, np_max, F_init, Low, Upp, xk_in]
    Upp = Upp.squeeze()
    Low = Low.squeeze()
    Upp = np.atleast_1d(Upp)
    Low = np.atleast_1d(Low)
    return [flag, X_0, np_max, F_init, Low, Upp, xk_in]
