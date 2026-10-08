import functools

import numpy as np


def _single_eval_Ffun(X, Ffun, n, m):
    """
    |pounders| is written to always call Ffun in batch mode.  This wrapper
    adapts a user-provided single-evaluation Ffun with m>1 to the batch mode
    interface.
    """
    assert (m > 1) and (X.ndim == 2) and (X.shape[1] == n)
    k = X.shape[0]
    F_batch = np.full((k, m), np.nan, float)
    for i in range(k):
        # Allow Ffun to return an m-element 1D array, 1 x m 2D array, or m x 1
        # 2D array.
        F_i = np.squeeze(Ffun(X[i, :]))
        if (F_i.ndim != 1) or (len(F_i) != m):
            raise ValueError(f"Ffun result cannot be converted into a {m}-element NumPy array")
        F_batch[i, :] = F_i
    return F_batch


def _single_eval_Ffun_m_eq_1(X, Ffun, n):
    """
    This is a special version of _single_eval_Ffun that handles just the m=1
    special case.
    """
    assert (X.ndim == 2) and (X.shape[1] == n)
    k = X.shape[0]
    F_batch = np.full((k, 1), np.nan, float)
    for i in range(k):
        # Allow Ffun to return a scalar, a single-element 1D array, or a 1x1 2D
        # array.
        F_i = np.array(Ffun(X[i, :]))
        if F_i.ndim not in (0, 1, 2):
            raise ValueError("Ffun result is not a scalar, 1D array, or 2D array")
        F_i = np.squeeze(F_i)
        if F_i.ndim != 0:
            raise ValueError("Ffun result does not correspond to a single value")
        F_batch[i, 0] = F_i
    return F_batch


def _batched_Ffun(X, Ffun, n, m):
    """
    This wraps batched Ffuns with m>1.  Since |pounders| is written to always
    call Ffun in batch mode, this wrapper is not strictly necessary.  However,
    instead of having |pounders| check that the user-provided Ffun has the
    correct batch interface on the first call of Ffun during execution, we let
    this wrapper check this on each call.  This results in more maintainable,
    clean code with acceptably small overhead.  For instance, developers do not
    need to ensure that they have correctly identified all possible first
    evaluations of Ffun.
    """
    assert (m > 1) and (X.ndim == 2) and (X.shape[1] == n)
    k = X.shape[0]
    F_batch = np.array(Ffun(X))
    if (F_batch.ndim != 2) or (F_batch.shape != (k, m)):
        raise ValueError(f"Ffun result is not a {k}x{m} array")
    return F_batch


def _batched_Ffun_m_eq_1(X, Ffun, n):
    """
    This is a special version of _batched_Ffun that handles just the m=1 special
    case.
    """
    assert (X.ndim == 2) and (X.shape[1] == n)
    k = X.shape[0]
    # Allow Ffun to return a k-element 1D array, kx1 2D array, or a 1xk 2D
    # array.
    F = np.array(Ffun(X))
    if F.ndim not in (1, 2):
        raise ValueError("Ffun result is not a 1D or 2D array")
    # Account for k=1 special case.
    F = np.atleast_1d(np.squeeze(F))
    if (F.ndim != 1) or (len(F) != k):
        raise ValueError(f"Ffun result cannot be converted into a {k}x1 NumPy array")
    return F.reshape((k, 1))


def force_Ffun_to_batched(Ffun, n, m, batched_Ffun):
    r"""
    Wrap the given Ffun so that its interface satisfies a single, batch-based
    interface.  Calling code
    * must always pass a :math:`k \times \np` 2D NumPy array to the wrapped
      Ffun, where :math:`k` is the number of points in the batch;
    * can assume that the wrapped Ffun always returns a :math:`k \times \nd` 2D
      NumPy array whose row ordering matches that of the input array; and
    * can assume that the wrapped Ffun will raise an exception if the
      user-provided Ffun's interface is invalid.

    The Ffun wrapper does **not** check if the contents of the input and output
    are valid.
    """
    assert m >= 1
    if batched_Ffun and (m == 1):
        return functools.partial(_batched_Ffun_m_eq_1, Ffun=Ffun, n=n)
    elif batched_Ffun and (m > 1):
        return functools.partial(_batched_Ffun, Ffun=Ffun, n=n, m=m)
    elif m == 1:
        return functools.partial(_single_eval_Ffun_m_eq_1, Ffun=Ffun, n=n)
    return functools.partial(_single_eval_Ffun, Ffun=Ffun, n=n, m=m)
