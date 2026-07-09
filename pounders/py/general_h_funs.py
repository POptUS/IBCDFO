"""
IMPORTANT: The set of functions provided here should match exactly the analogous
set of functions offered in the MATLAB POUNDERS implementation.  In addition,
the inline documentation in this file should be correct for both sets of
functions.
"""

import numpy as np


def h_identity(F):
    r"""
    Identity :math:`\hfun` function for using |pounders| when the objective is
    not composite; that is, when :math:`\Ffun: \R^{\np} \to \R` is scalar-valued and

    .. math::

        f(\psp) = \hfun\left(\Ffun(\psp)\right) = \Ffun(\psp).

    When using this :math:`\hfun` function, the ``combine_identity`` function
    should also be passed to |pounders|.
    """
    return np.squeeze(F)


def combine_identity(Cres, Gres, Hres):
    return Gres.squeeze(), Hres.squeeze()


def h_neg_leastsquares(F):
    r"""
    :math:`\hfun` function for constructing the negative
    least-squares objective function

    .. math::

        f(\psp) = \hfun\left(\Ffun(\psp)\right)
                = -\sum_{i = 1}^{\nd} \Ffuncomp{i}(\psp)^2.

    The ``combine_neg_leastsquares`` function should also be passed to
    |pounders| when using this :math:`\hfun` function.
    """
    return -h_leastsquares(F)


def combine_neg_leastsquares(Cres, Gres, Hres):
    G, H = combine_leastsquares(Cres, Gres, Hres)
    return -G, -H


def h_leastsquares(F):
    r"""
    :math:`\hfun` function for constructing the standard
    least-squares objective function

    .. math::

        f(\psp) = \hfun\left(\Ffun(\psp)\right)
                = \sum_{i = 1}^{\nd} \Ffuncomp{i}(\psp)^2,

    which is the :math:`\hfun` function used by default.

    The ``combine_leastsquares`` function should also be passed to |pounders|
    when using this :math:`\hfun` function.
    """
    return np.sum(F**2)


def combine_leastsquares(Cres, Gres, Hres):
    Cres = np.asarray(Cres, dtype=float).reshape(-1)
    Gres = np.asarray(Gres, dtype=float)

    G = 2 * Gres @ Cres
    H = 2 * Gres @ Gres.T
    return G, H


def combine_emittance(Cres, Gres, Hres):
    n, _, m = Hres.shape

    assert m == 3, "Emittance calculation requires exactly three quantities"

    G = Cres[0] * Gres[:, 1] + Cres[1] * Gres[:, 0] - 2 * Cres[2] * Gres[:, 2]
    H = Cres[0] * Hres[:, :, 1] + Cres[1] * Hres[:, :, 0] + np.outer(Gres[:, 1], Gres[:, 0]) + np.outer(Gres[:, 0], Gres[:, 1]) - 2 * Cres[2] * Hres[:, :, 2] - 2 * np.outer(Gres[:, 2], Gres[:, 2])

    return G, H


def h_emittance(F):
    r"""
    :math:`\hfun` function for constructing the emittance objective
    function

    .. math::

        f(\psp) = \hfun\left(\Ffun(\psp)\right)
                = \Ffuncomp{1}(\psp)\Ffuncomp{2}(\psp) - \Ffuncomp{3}(\psp)^2

    limited to the special case of :math:`\Ffun : \R^{\np} \to \R^3`.

    The ``combine_emittance`` function should also be passed to |pounders|
    when using this :math:`\hfun` function.
    """
    assert len(F) == 3, "Emittance must have exactly 3 inputs"
    return F[0] * F[1] - F[2] ** 2


def h_loglikelihood(probabilities, counts):
    r"""
    Multinomial log-likelihood for outcome probabilities and observed counts.

    This returns the data-dependent part of

    .. math::

        \log L(p; n) = \sum_i n_i \log(p_i),

    omitting multinomial constants that do not depend on ``probabilities``.
    The return value is a scalar.  To use this as a minimization objective,
    minimize ``-h_loglikelihood(probabilities, counts)``.
    """
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    counts = np.asarray(counts, dtype=float).reshape(-1)

    if probabilities.shape != counts.shape:
        raise ValueError(
            "probabilities and counts must have the same flattened shape."
        )
    if np.any(counts < 0):
        raise ValueError("counts must be nonnegative.")

    safe_probabilities = np.clip(probabilities, 1e-15, 1.0)
    return float(np.sum(counts * np.log(safe_probabilities)))


def h_negative_loglikelihood(probabilities, counts):
    r"""
    Negative multinomial log-likelihood for minimization.

    This is the minimization-ready version of ``h_loglikelihood``:

    .. math::

        -\log L(p; n) = -\sum_i n_i \log(p_i).
    """
    return -h_loglikelihood(probabilities, counts)


def combine_negative_loglikelihood(Cres, Gres, Hres, info=None):
    r"""
    Combine probability models for the negative multinomial log-likelihood.

    This builds a local model for

    .. math::

        -\log L(p; n) = -\sum_i n_i \log(p_i).

    ``Gres`` is expected to be the Jacobian of the probabilities with shape
    ``(n_parameters, n_probability_entries)``.  The probabilities and counts
    are read from ``info["p"]`` and ``info["counts"]``.

    When ``Hres`` contains per-entry Hessians, this includes the exact
    second-derivative term.  When ``Hres`` is the implicit-zero tuple used by
    the GST notebook, this returns the Gauss-Newton/Fisher approximation.
    """
    if info is None or "p" not in info or "counts" not in info:
        raise ValueError(
            "combine_negative_loglikelihood requires info with 'p' and 'counts'."
        )

    probabilities = np.asarray(info["p"], dtype=float).reshape(-1)
    counts = np.asarray(info["counts"], dtype=float).reshape(-1)
    Gres = np.asarray(Gres, dtype=float)

    if probabilities.shape != counts.shape:
        raise ValueError("probabilities and counts must have the same shape.")
    if Gres.shape[1] != probabilities.size:
        raise ValueError(
            "Gres must have one column per probability/count entry."
        )
    if np.any(counts < 0):
        raise ValueError("counts must be nonnegative.")

    safe_probabilities = np.clip(probabilities, 1e-15, 1.0)

    grad_coefficients = -counts / safe_probabilities
    hessian_diagonal = counts / (safe_probabilities**2)

    G = Gres @ grad_coefficients
    H = (Gres * hessian_diagonal) @ Gres.T

    if not isinstance(Hres, tuple):
        Hres = np.asarray(Hres, dtype=float)
        if Hres.ndim == 3 and Hres.shape[2] == probabilities.size:
            for i in range(probabilities.size):
                H = H + grad_coefficients[i] * Hres[:, :, i]

    return G, H
