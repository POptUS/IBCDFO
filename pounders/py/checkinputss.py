import numbers

import numpy as np


def checkinputss(Ffun, X_0, n, np_max, nf_max, g_tol, delta_0, nfs, m, X_init, F_init, xk_in, Low, Upp):
    """
    Raise an exception if any of the given POUNDERS specification variables do
    not meet the requirements of the POUNDERS implementation.

    For information regarding the arguments, please see the documentation for
    :py:func:`ibcdfo.pounders.pounders`.
    """
    # IMPORTANT: In error messages, use the name of the argument as passed to
    # pounders.  The checks performed here should be maintained consistent with
    # the specifications provided for these arguments in the pounders.py inline
    # docs and general POUNDERS documentation.

    # Check Ffun
    # To avoid unnecessary function evaluations, we do not confirm here that the
    # function accepts real vectors of the correct shape nor that it returns
    # real vectors of the correct shape.  It is assumed that the POUNDERS
    # implementation performs that error checking.
    if not callable(Ffun):
        raise TypeError("Error: Ffun is not a function")

    # Problem dimensions
    if not isinstance(n, numbers.Integral):
        raise TypeError(f"Error: dimension n is not an integer ({n})")
    if n < 1:
        raise ValueError(f"Error: dimension n is not a positive integer ({n})")

    if not isinstance(m, numbers.Integral):
        raise TypeError(f"Error: dimension m is not an integer ({m})")
    if m < 1:
        raise ValueError(f"Error: dimension m is not a positive integer ({m})")

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
    if any(Upp <= Low):
        raise ValueError("Error: must have Upp > Low")

    # Check starting point - must be finite and in feasible set
    if not isinstance(X_0, np.ndarray):
        raise TypeError("Error: X_0 must be a 1D NumPy array")
    if (X_0.ndim != 1) or (len(X_0) != n):
        raise ValueError(f"Error: X_0 is not an {n}-element 1D NumPy array")
    if (not np.all(np.isfinite(X_0))) or (not np.all(np.isreal(X_0))):
        raise ValueError("Error: X_init must contain only finite, real values")
    if any(X_0 < Low) or any(X_0 > Upp):
        raise ValueError("Error: X_0 outside of Low/Upp bounds")

    # Check max number of interpolation points
    if not isinstance(np_max, numbers.Integral):
        raise TypeError(f"Error: np_max is not an integer ({np_max})")
    if (np_max < n + 1) or (np_max > int(0.5 * (n + 1) * (n + 2))):
        raise ValueError(f"Error: np_max ({np_max}) is not in [n+1, 0.5 * (n+1) * (n+2)]")

    # Check standard positive quantities
    if not isinstance(nfs, numbers.Integral):
        raise TypeError(f"Error: nfs is not an integer ({nfs})")
    if nfs < 0:
        raise ValueError(f"Error: nfs is not a positive integer ({nfs})")

    # nf_max is the max actual evaluations to be allowed during an optimization.
    # It does not include any preexisting evaluations provided to POUNDERS.
    #
    # There is no sense in running an optimization without evaluating at least
    # once at a non-geometry point chosen by POUNDERS.  If users provide any
    # number of preexisting evaluations, there's no guarantee that any of those
    # values (other than at the starting point) will be used by POUNDERS to
    # start the optimization.  Even if they did, we still want POUNDERS to
    # perform one evaluation at a point determined by POUNDERS using those
    # values.
    #
    # We, therefore, establish a lower bound on nf_max assuming that users can
    # only know with certainty that a given starting point value can be used by
    # POUNDERS.  They must assume that all other provided values, if any, might
    # be ignored.
    min_required_evals = n + 1 if nfs > 0 else n + 2
    if not isinstance(nf_max, numbers.Integral):
        raise TypeError(f"Error: nf_max is not an integer ({nf_max})")
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

    # Check prior evaluations
    if not isinstance(xk_in, numbers.Integral):
        raise TypeError(f"Error: xk_in is not an integer ({xk_in})")
    elif (xk_in < 0) or ((nfs == 0) and (xk_in != 0)) or ((nfs > 0) and (xk_in >= nfs)):
        raise ValueError(f"Error: Invalid xk_in ({xk_in})")

    # As per the docs, we allow all non-X_init[xk_in, :] points to be infeasible.
    if not isinstance(X_init, np.ndarray):
        raise TypeError("Error: X_init must be a 2D NumPy array")
    if X_init.ndim != 2:
        raise ValueError("Error: X_init must be a 2D NumPy array")
    if X_init.shape != (nfs, n):
        raise ValueError(f"Error: X_init has shape {X_init.shape} instead of ({nfs}, {n})")
    if nfs > 0:
        if (not np.all(np.isfinite(X_init))) or (not np.all(np.isreal(X_init))):
            raise ValueError("Error: X_init must contain only finite, real values")
        if not np.array_equal(X_init[xk_in, :], X_0, equal_nan=False):
            raise ValueError("Error: X_0 doesn't match X_init[xk_in, :]")

        # While one could argue that including a point more than once in X_init
        # could be acceptable so long as their F_init values were identical, we
        # prefer to consider it as a logic error in calling code.  We,
        # therefore, inform calling code explicitly of this issue (rather than
        # post a warning or fix it for them) to allow them to assess why this
        # error was made and fix it.
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
        if (not np.all(np.isfinite(F_init))) or (not np.all(np.isreal(F_init))):
            raise ValueError("Error: F_init must contain only finite, real values")
