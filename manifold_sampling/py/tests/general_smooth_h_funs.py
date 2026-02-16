import numpy as np


def h_leastsquares(z, H0=None):
    r"""
    :math:`\hfun` function for constructing the standard least-squares objective
    function

    .. math::

        f(\psp) = \hfun\left(\Ffun(\psp)\right)
                = \sum_{i = 1}^{\nd} \Ffuncomp{i}(\psp)^2.

    It is intended that this be used only for testing purposes.
    """
    n = len(z)
    h = np.sum(z**2)
    grads = np.reshape(2.0 * z, (n, 1))

    if H0 is None:
        return h, grads, [str(1)]

    return np.array([h]), grads
