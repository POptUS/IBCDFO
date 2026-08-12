import numbers

import numpy as np


def checkinputss(Ffun, X_0, n, np_max, nf_max, g_tol, delta_0, nfs, m, X_init, F_init, xk_in, Low, Upp):
    """
    THIS DOES NOT ALTER ANY OF THE INPUT ARGUMENTS.

    Confirm that inputs provided to POUNDERS achieve the expectations of the
    POUNDERS implementation and print warning and error messages where
    applicable.

    For information regarding the arguments, please the documentation for
    :py:func:`ibcdfo.pounders.pounders`.

    :return:
        * 1 if all arguments are valid
        * 0 if a warning was printed
        * -1 if an invalid argument was given
    """
    # IMPORTANT: In error messages, use the name of the argument as passed to
    # pounders.
    #
    # While we could run every check regardless of any failures so that calling
    # code can immediately find all mistakes, that would substantially decrease
    # the utility of our unit test, which doesn't check the actual error
    # message to identify what error occurred.

    # ----- HARDCODED VALUES
    SUCCESS = 1
    ERROR = -1

    # ----- ERROR CHECKING
    flag = SUCCESS

    # Check Ffun
    if not callable(Ffun):
        print("Error: Ffun is not a function handle")
        return ERROR

    # Problem dimensions
    if not isinstance(n, numbers.Integral):
        print(f"Error: n dimension is not an integer ({n})")
        return ERROR
    if n < 1:
        print(f"Error: n dimension is not positive integer ({n})")
        return ERROR

    if not isinstance(m, numbers.Integral):
        print(f"Error: m dimension is not an integer ({m})")
        return ERROR
    if m < 1:
        print(f"Error: m dimension is not positive integer ({m})")
        return ERROR

    # Check max number of interpolation points
    if not isinstance(np_max, numbers.Integral):
        print(f"Error: np_max is not an integer ({np_max})")
        return ERROR
    if np_max < n + 1 or np_max > int(0.5 * (n + 1) * (n + 2)):
        np_max = max(n + 1, min(np_max, int(0.5 * (n + 1) * (n + 2))))
        print("Error: np_max not in [n+1, 0.5 * (n+1) * (n+2)]")
        return ERROR

    # Check standard positive quantities
    if not isinstance(nfs, numbers.Integral):
        print(f"Error: nfs not an integer ({nfs})")
        return ERROR
    if nfs < 0:
        print(f"Error: nfs is not positive integer ({nfs})")
        return ERROR

    # nf_max is the actual evaluations to be made during the optimization.
    # There is no sense in running the optimizatio without at least one
    # evaluation beyond the n+1 geometry points.
    if not isinstance(nf_max, numbers.Integral):
        print(f"Error: nf_max is not an integer ({nf_max})")
        return ERROR
    min_required_evals = np.max(((n + 1) - nfs, 1))
    if nf_max < min_required_evals:
        print(f"Error: max number of evaluations ({nf_max}) should be >= {min_required_evals}")
        return ERROR

    if (not np.isreal(g_tol)) or (not np.isfinite(g_tol)):
        print(f"Error: g_tol is not a finite real ({g_tol})")
        return ERROR
    if g_tol <= 0.0:
        print("Error: g_tol must be positive")
        return ERROR

    if (not np.isreal(delta_0)) or (not np.isfinite(delta_0)):
        print(f"Error: delta_0 is not a finite real ({delta_0})")
        return ERROR
    if delta_0 <= 0.0:
        print("Error: delta_0 must be positive")
        return ERROR

    # Check the bounds
    if not isinstance(Low, np.ndarray):
        print("Error: Low must be Numpy array")
        return ERROR
    if Low.ndim != 1:
        print("Error: Low must be 1D Numpy array")
        return ERROR
    if len(Low) != n:
        print(f"Error: Low is not {n} element 1D Numpy array")
        return ERROR
    if np.any(np.isnan(Low)):
        print("Error: Low contain non-finite values")
        return ERROR
    if not isinstance(Upp, np.ndarray):
        print("Error: Upp must be Numpy array")
        return ERROR
    if Upp.ndim != 1:
        print("Error: Upp must be 1D Numpy array")
        return ERROR
    if len(Upp) != n:
        print(f"Error: Upp is not {n} element 1D Numpy array")
        return ERROR
    if np.any(np.isnan(Upp)):
        print("Error: Upp contain non-finite values")
        return ERROR
    if any(Low >= Upp):
        print("Error: must have Upp > Low")
        return ERROR

    # Check starting point
    if not isinstance(X_0, np.ndarray):
        print("Error: X_0 must be a Numpy array")
        return ERROR
    elif (X_0.ndim != 1) or (len(X_0) != n):
        print(f"Error: X_0 is not an {n} element 1D Numpy array")
        return ERROR
    if any(X_0 < Low) or any(X_0 > Upp):
        print("Error: starting point outside of bounds (Low,Upp)")
        return ERROR

    # Check prior evaluations
    if not isinstance(xk_in, numbers.Integral):
        print("Error: starting point index not an integer")
        return ERROR
    elif (xk_in < 0) or ((nfs == 0) and (xk_in != 0)) or ((nfs > 0) and (xk_in >= nfs)):
        print("Error: Invalid starting point index")
        return ERROR

    if X_init.ndim != 2:
        print("Error: X_init must be 2D arrays")
        return ERROR
    if X_init.shape[0] != nfs:
        print("Error: number of initial points nfs does not match input X_init")
        return ERROR
    elif X_init.shape[1] != n:
        print("Error: X_init does not contain the right number of coordinates")
        return ERROR
    if nfs > 0:
        if not np.all(np.isfinite(X_init)):
            print("Error: X_init contains non-finite values")
            return ERROR
        if not np.array_equiv(X_init[xk_in], X_0):
            print("Error: Starting point X_0 doesn't match row in X_init[xk_in]")
            return ERROR

    if F_init.ndim != 2:
        print("Error: F_init must be 2D arrays")
        return ERROR
    if F_init.shape[0] != nfs:
        print("Error: number of starting f values nfs does not match input F_init")
        return ERROR
    elif F_init.shape[1] != m:
        print("Error: F_init does not contain the right number of residuals")
        return ERROR
    if nfs > 0:
        if not np.all(np.isfinite(F_init)):
            print("Error: F_init contains non-finite values")
            return ERROR

    return flag
