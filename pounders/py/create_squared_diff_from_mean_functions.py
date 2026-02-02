import numbers

import numpy as np


def create_squared_diff_from_mean_functions(alpha):
    r"""
    Create an :math:`\hfun` function and its associated combinemodel function,
    both using the given parameter value :math:`\alpha`, for constructing and
    using the |pounders| objective function

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
    :return: (hfun, combinemodels) constructed with the given :math:`\alpha`
    """
    # NOTE: Don't alter alpha anywhere in this function.

    if not isinstance(alpha, numbers.Real):
        raise TypeError("alpha parameter must be floating point number")
    elif not np.isfinite(alpha):
        raise ValueError("alpha parameter must be finite real")

    # Both of these definitions assume that alpha is a variable in the local
    # scope of this function.  In this case, it's the function's argument.
    def hfun(F):
        F_avg = np.mean(F)
        return np.sum((F - F_avg) ** 2) - alpha * F_avg**2

    def combinemodels(Cres, Gres, Hres):
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

    # The value of the given actual alpha argument is fixed into the returned
    # functions only at this point.
    return hfun, combinemodels
