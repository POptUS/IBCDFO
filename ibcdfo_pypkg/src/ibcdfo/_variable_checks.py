"""
.. todo::
    * Determine if other loopholes should be accounted for or at least
      documented
    * These should eventually be moved to the ``poptus`` package
"""

import numbers

import numpy as np


def is_integer(n):
    """
    :return: ``True`` if **n** is of an integer type.  Aside from the obvious
        cases, ``False`` is returned if **n** is a ``bool`` or if **n** is an
        integer value stored in floating point format (|eg| 1.0).
    """
    return isinstance(n, numbers.Integral) and (not isinstance(n, bool))


def is_finite_real(x):
    """
    :return: ``True`` if **x** is a floating point variable whose content is a
        finite real.  Aside from the obvious cases, ``False`` is returned if
        **x** is ``NaN``, ``+/-Inf``, a ``bool``, a complex variable with or
        without an imaginay part, or, for example, any container containing a
        finite real.
    """
    return isinstance(x, numbers.Real) and (not isinstance(x, bool)) and \
        (not np.iscomplexobj(x)) and np.isfinite(x)  # fmt: skip


def is_finite_real_numpy_array(x, ndim):
    """
    :return: ``True`` if **x** is a NumPy array of dimension **ndim** whose
        elements are all finite real.  Aside from the obvious cases, ``False``
        is returned if any of the elements are ``NaN``, ``+/-Inf``, or complex
        variables with or without imaginary parts.  It also returns ``False``
        even if the array could be correctly squeezed or extended to **ndim**.
    """
    assert ndim >= 1
    return isinstance(x, np.ndarray) and (x.ndim == ndim) and \
        np.issubdtype(x.dtype, np.floating) and np.isreal(x).all() and \
        (not np.iscomplexobj(x)) and np.isfinite(x).all()   # fmt: skip


def is_extended_real_numpy_array(x, ndim):
    """
    :return: ``True`` if **x** is a NumPy array of dimension **ndim** whose
        elements are all in the extended reals.  Aside from the obvious cases,
        ``False`` is returned if any of the elements are ``NaN`` or complex
        variables with or without imaginary parts.  It also returns ``False``
        even if the array could be correctly squeezed or extended to **ndim**.
    """
    assert ndim >= 1
    return isinstance(x, np.ndarray) and (x.ndim == ndim) and \
        np.issubdtype(x.dtype, np.floating) and np.isreal(x).all() and \
        (not np.iscomplexobj(x)) and (not np.isnan(x).any())  # fmt: skip
