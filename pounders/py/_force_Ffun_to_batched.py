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


def _single_eval_Ffun_scalar(X, Ffun, n):
    """
    This is a special version of _single_eval_Ffun that directly handles just
    the m=1 special case.
    """
    assert (X.ndim == 2) and (X.shape[1] == n)
    k = X.shape[0]
    F_batch = np.full((k, 1), np.nan, float)
    for i in range(k):
        # Allow Ffun to return a scalar, 1 single-element 1D array, or a 1x1 2D
        # array.
        F_i = np.atleast_1d(np.squeeze(Ffun(X[i, :])))
        if (F_i.ndim != 1) or (len(F_i) != 1):
            raise ValueError("Ffun result cannot be converted into a single-element NumPy array")
        F_batch[i, :] = F_i
    return F_batch


def _batched_Ffun(X, Ffun, n, m):
    """
    Since |pounders| is written to always call Ffun in batch mode, this wrapper
    is not strictly necessary.  However, instead of having |pounders| check that
    the user-provided Ffun has the correct batch interface on the first call of
    Ffun during execution, we let this wrapper check this on each call.  This
    results in more maintainable, clean optimization code with acceptably small
    overhead.  For instance, developers do not need to ensure that they have
    correctly identified all possible first evaluations of Ffun.
    """
    assert (m > 1) and (X.ndim == 2) and (X.shape[1] == n)
    k = X.shape[0]
    F_batch = np.array(Ffun(X), copy=False)
    if (F_batch.ndim != 2) or (F_batch.shape != (k, m)):
        raise ValueError(f"Ffun result cannot be converted into a {k}x{m} NumPy array")
    return F_batch


def _batched_Ffun_scalar(X, Ffun, n):
    """
    This is a special version of _batched_Ffun that directly handles just
    the m=1 special case.
    """
    assert (X.ndim == 2) and (X.shape[1] == n)
    k = X.shape[0]
    F = np.atleast_1d(np.squeeze(Ffun(X)))
    if (F.ndim != 1) or (len(F) != k):
        raise ValueError(f"Ffun result cannot be converted into a {k}x1 NumPy array")
    return F.reshape((k, 1))


def force_Ffun_to_batched(Ffun, n, m, batched_Ffun):
    r"""
    Wrap the given Ffun so that its interface satisfies the single, batch-based
    interface specified for and used consistently by the |pounders|
    implementation.

    Calling code
    * must always pass a :math:`k \times \np` 2D NumPy array to the wrapped Ffun
    * can assume that the wrapped Ffun always returns a :math:`k \times \nd` 2D
      NumPy array whose row ordering matches that of the input array
    * can assume that the wrapped Ffun will raise an exception if the
      user-provided Ffun's interface is not valid for its stated evaluation
      type.

    The Ffun wrapper does **not** check if the contents of the input and output
    are valid.
    """
    if batched_Ffun and (m == 1):
        return functools.partial(_batched_Ffun_scalar, Ffun=Ffun, n=n)
    elif batched_Ffun and (m > 1):
        return functools.partial(_batched_Ffun, Ffun=Ffun, n=n, m=m)
    elif m == 1:
        return functools.partial(_single_eval_Ffun_scalar, Ffun=Ffun, n=n)
    return functools.partial(_single_eval_Ffun, Ffun=Ffun, n=n, m=m)
