"""
IMPORTANT: The set of functions provided here should match exactly the analogous
set of functions offered in the MATLAB POUNDERS implementation.  In addition,
the inline documentation in this file should be correct for both sets of
functions.
"""

import numpy as np


def h_identity(F):
    r"""
    :math:`\hfun` function that allows users to use |pounders| for the special
    case that their :math:`\Ffun: \R^{\np} \to \R` is already an objective
    function or

    .. math::

        f(\psp) = \hfun\left(\Ffun(\psp)\right) = \Ffun(\psp).

    The ``combine_identity`` function should also be passed to |pounders| when
    using this :math:`\hfun` function.
    """
    return np.squeeze(F)


def combine_identity(Cres, Gres, Hres):
    return Gres.squeeze(), Hres.squeeze()


def h_neg_leastsquares(F):
    r"""
    :math:`\hfun` function for constructing the |pounders| negative
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
    :math:`\hfun` function for constructing the standard |pounders|
    least-squares objective function

    .. math::

        f(\psp) = \hfun\left(\Ffun(\psp)\right)
                = \sum_{i = 1}^{\nd} \Ffuncomp{i}(\psp)^2,

    which is the :math:`\hfun` function used be default.

    The ``combine_leastsquares`` function should also be passed to |pounders|
    when using this :math:`\hfun` function.
    """
    return np.sum(F**2)


def combine_leastsquares(Cres, Gres, Hres):
    n, _, m = Hres.shape

    G = 2 * Gres @ Cres.T
    H = np.zeros((n, n))
    for i in range(m):
        H = H + Cres[i] * Hres[:, :, i]

    H = 2 * H + 2 * Gres @ Gres.T

    return G, H


def combine_emittance(Cres, Gres, Hres):
    n, _, m = Hres.shape

    assert m == 3, "Emittance calculation requires exactly three quantities"

    G = Cres[0] * Gres[:, 1] + Cres[1] * Gres[:, 0] - 2 * Cres[2] * Gres[:, 2]
    H = Cres[0] * Hres[:, :, 1] + Cres[1] * Hres[:, :, 0] + np.outer(Gres[:, 1], Gres[:, 0]) + np.outer(Gres[:, 0], Gres[:, 1]) - 2 * Cres[2] * Hres[:, :, 2] - 2 * np.outer(Gres[:, 2], Gres[:, 2])

    return G, H


def h_emittance(F):
    r"""
    :math:`\hfun` function for constructing the |pounders| emittance objective
    function

    .. math::

        f(\psp) = \hfun\left(\Ffun(\psp)\right)
                = \Ffuncomp{1}(\psp)\Ffuncomp{2}(\psp) - \Ffuncomp{3}(\psp)

    limited to the special case of :math:`\Ffun : \R^{\np} \to \R^3`.

    The ``combine_emittance`` function should also be passed to |pounders|
    when using this :math:`\hfun` function.
    """
    assert len(F) == 3, "Emittance must have exactly 3 inputs"
    return F[0] * F[1] - F[2] ** 2


def h_squared_diff_from_mean(F, alpha):
    r"""
    :math:`\hfun` function for constructing the |pounders| objective function

    .. math::

        f(\psp; \alpha) = \hfun\left(\Ffun(\psp); \alpha\right)
          = \sum_{i=1}^{\nd} \left(\Ffuncomp{i}(\psp) - \overline{\Ffun}(\psp)\right)^2
            - \alpha \overline{\Ffun}(\psp)^2

    where

    .. math::

        \overline{\Ffun}(\psp) = \frac{1}{\nd}\sum_{i=1}^{\nd} \Ffuncomp{i}(\psp)

    is the average value of all components in :math:`\Ffun(\psp)`.   This
    objective, therefore, prefers vectors close to their average.

    :param: :math:`\alpha` is a problem-specific parameter that specifies how
        much the objective function should penalize small (large) averages for
        :math:`\alpha` positive (negative).

    Users can create |pounders|-compatible versions of this function and its
    combine models function for a particular :math:`\alpha` value with code such
    as

    .. code:: python

        import functools
        ALPHA = X.Y
        hfun = functools.partial(h_squared_diff_from_mean, alpha=ALPHA)
        combinemodels = functools.partial(combine_squared_diff_from_mean, alpha=ALPHA)
    """
    F_avg = np.mean(F)
    return np.sum((F - F_avg) ** 2) - alpha * F_avg**2


def combine_squared_diff_from_mean(Cres, Gres, Hres, alpha):
    n, _, m = Hres.shape

    m_sumF = np.mean(Cres)
    m_sumG = 1 / m * np.sum(Gres, axis=1)
    sumH = np.sum(Hres, axis=2)

    G = np.zeros(n)
    for i in range(m):
        G = G + (Cres[i] - m_sumF) * (Gres[:, i] - m_sumG)
    G = 2 * G - 2 * alpha * m_sumF * m_sumG

    H = np.zeros((n, n))
    for i in range(m):
        H = H + (Cres[i] - m_sumF) * (Hres[:, :, i] + sumH) + np.outer(Gres[:, i] - m_sumG, Gres[:, i] - m_sumG)

    H = 2 * H

    H = H - (2 * alpha / m) * m_sumF * sumH - (2 * alpha) * np.outer(m_sumG, m_sumG)

    return G, H
