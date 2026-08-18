import numbers

import numpy as np


def checkinputss(Ffun, X_0, n, np_max, nf_max, g_tol, delta_0, nfs, m, X_init, F_init, xk_in, Low, Upp):
    """
    Raise an exception if any of the given POUNDERS specification variables do
    not meet the expectations of the POUNDERS implementation.

    For information regarding the arguments, please see the documentation for
    :py:func:`ibcdfo.pounders.pounders`.
    """
    # IMPORTANT: In error messages, use the name of the argument as passed to
    # pounders.  The checks performed here should be maintained consistent with
    # the specifications provided for these arguments in the pounders.py docs.

    # Check Ffun
    if not callable(Ffun):
        raise TypeError("Error: Ffun is not a function handle")

    # Problem dimensions
    if not isinstance(n, numbers.Integral):
        raise TypeError(f"Error: dimension n is not an integer ({n})")
    if n < 1:
        raise ValueError(f"Error: dimension n is not a positive integer ({n})")

    if not isinstance(m, numbers.Integral):
        raise TypeError(f"Error: dimension m is not an integer ({m})")
    if m < 1:
        raise ValueError(f"Error: dimension m is not a positive integer ({m})")

    # Check max number of interpolation points
    if not isinstance(np_max, numbers.Integral):
        raise TypeError(f"Error: np_max is not an integer ({np_max})")
    if np_max < n + 1 or np_max > int(0.5 * (n + 1) * (n + 2)):
        raise ValueError(f"Error: np_max ({np_max}) is not in [n+1, 0.5 * (n+1) * (n+2)]")

    # Check standard positive quantities
    if not isinstance(nfs, numbers.Integral):
        raise TypeError(f"Error: nfs is not an integer ({nfs})")
    if nfs < 0:
        raise ValueError(f"Error: nfs is not a positive integer ({nfs})")

    # nf_max is the max actual evaluations to be made during an optimization.
    # There is no sense in running an optimization without evaluating at least
    # once at a non-geometry point chosen by POUNDERS.  Even if calling code
    # provides more than n+1 values, we still let POUNDERS choose one point.
    if not isinstance(nf_max, numbers.Integral):
        raise TypeError(f"Error: nf_max is not an integer ({nf_max})")
    min_required_evals = np.max((n - nfs + 2, 1))
    if nf_max < min_required_evals:
        raise ValueError(f"Error: nf_max ({nf_max}) should be >= {min_required_evals}")

    if not isinstance(g_tol, numbers.Real):
        raise TypeError(f"Error: g_tol is not a real ({g_tol})")
    if not np.isfinite(g_tol):
        raise ValueError(f"Error: g_tol is not a finite real ({g_tol})")
    if g_tol <= 0.0:
        raise ValueError(f"Error: g_tol must be a positive real ({g_tol})")

    if not isinstance(delta_0, numbers.Real):
        raise TypeError(f"Error: delta_0 is not a real ({delta_0})")
    if not np.isfinite(delta_0):
        raise ValueError(f"Error: delta_0 is not a finite real ({delta_0})")
    if delta_0 <= 0.0:
        raise ValueError(f"Error: delta_0 must be a positive real ({delta_0})")

    # Check the bounds
    if not isinstance(Low, np.ndarray):
        raise TypeError("Error: Low must be a 1D NumPy array")
    if (Low.ndim != 1) or (len(Low) != n):
        raise ValueError(f"Error: Low is not an {n}-element 1D NumPy array")
    if np.any(np.isnan(Low)):
        raise ValueError("Error: Low contains NaN values")
    if not isinstance(Upp, np.ndarray):
        raise TypeError("Error: Upp must be a 1D NumPy array")
    if (Upp.ndim != 1) or (len(Upp) != n):
        raise ValueError(f"Error: Upp is not an {n}-element 1D NumPy array")
    if np.any(np.isnan(Upp)):
        raise ValueError("Error: Upp contains NaN values")
    if any(Low >= Upp):
        raise ValueError("Error: must have Upp > Low")

    # Check starting point
    if not isinstance(X_0, np.ndarray):
        raise TypeError("Error: X_0 must be a 1D NumPy array")
    elif (X_0.ndim != 1) or (len(X_0) != n):
        raise ValueError(f"Error: X_0 is not an {n}-element 1D NumPy array")
    if any(X_0 < Low) or any(X_0 > Upp):
        raise ValueError("Error: starting point outside of Low/Upp bounds")

    # Check prior evaluations
    if not isinstance(xk_in, numbers.Integral):
        raise TypeError(f"Error: starting point index is not an integer ({xk_in})")
    elif (xk_in < 0) or ((nfs == 0) and (xk_in != 0)) or ((nfs > 0) and (xk_in >= nfs)):
        raise ValueError(f"Error: Invalid starting point index ({xk_in})")

    if not isinstance(X_init, np.ndarray):
        raise TypeError("Error: X_init must be a 2D NumPy array")
    if X_init.ndim != 2:
        raise ValueError("Error: X_init must be a 2D NumPy array")
    if X_init.shape != (nfs, n):
        raise ValueError(f"Error: X_init has shape {X_init.shape} instead of ({nfs}, {n})")
    if nfs > 0:
        if not np.all(np.isfinite(X_init)):
            raise ValueError("Error: X_init contains non-finite values")
        if not np.array_equiv(X_init[xk_in], X_0):
            raise ValueError("Error: X_0 doesn't match X_init[xk_in, :]")

        _, counts = np.unique(
            X_init,
            axis=0,
            return_index=False,
            return_inverse=False,
            return_counts=True,
        )
        if any(counts != 1):
            raise ValueError("Error: X_init contains repeated points")

    if not isinstance(F_init, np.ndarray):
        raise TypeError("Error: F_init must be a 2D NumPy array")
    if F_init.ndim != 2:
        raise ValueError("Error: F_init must be a 2D NumPy array")
    if F_init.shape != (nfs, m):
        raise ValueError(f"Error: Invalid F_init shape {F_init.shape}.  Expected ({nfs}, {m})")
    if nfs > 0:
        if not np.all(np.isfinite(F_init)):
            raise ValueError("Error: F_init contains non-finite values")
