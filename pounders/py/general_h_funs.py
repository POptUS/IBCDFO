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
