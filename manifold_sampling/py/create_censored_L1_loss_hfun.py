from itertools import product

import numpy as np


def create_censored_L1_loss_hfun(C, D):
    r"""
    Censored (one-sided) composite objective.  This is a generalized version of
    Womersley's censored :math:`\ell_1` loss function :cite:t:`womersley1986`.

    Given observed system outputs :math:`\zvec\in\R^m`, a per-component censoring
    floor :math:`\cvec\in\R^m`, and target data :math:`\dvec\in\R^m`, this
    objective is

    .. math::

        h(\zvec;\cvec,\dvec)
        = \sum_{i=1}^{m} \left|\, d_i - \max(z_i, c_i)\,\right|.

    This produces a one-sided (censored) discrepancy: components with
    :math:`z_i < c_i` are treated as if the observation were :math:`c_i`, so the
    loss does not continue to decrease by driving :math:`z_i` below the censoring
    floor. This reduces sensitivity to outliers, preventing any single component
    from dominating the measure of misfit.

    :param C: 1D numpy array of length :math:`m` that provides the censoring
        values :math:`c_i`.
    :param D: 1D numpy array of length :math:`m` that provides the target data
        :math:`d_i`.
    :return: ``hfun`` constructed with the given :math:`c_i, d_i` that is
        compatible only with :math:`\zvec` arguments of the same length as ``C``
        and ``D``.
    """
    # ----- CREATE LOCAL-SCOPE COPIES
    # This prevents the code here and in hfun from accidentally altering the
    # contents of the actual arrays passed into the function.
    #
    # IMPORTANT: Aside from making copies don't alter C or D anywhere in this
    # function.
    if not isinstance(C, np.ndarray):
        raise TypeError("C is not a numpy array")
    C = np.atleast_1d(np.squeeze(C.copy()))

    if not isinstance(D, np.ndarray):
        raise TypeError("D is not a numpy array")
    D = np.atleast_1d(np.squeeze(D.copy()))

    # ----- ERROR CHECK ARGUMENTS
    if C.ndim != 1:
        raise ValueError("C is not a vector")
    elif len(C) <= 1:
        # While C, D, and z that each contain only a single element are sensible
        # in terms of the math, there is presently no legitimate use case to
        # motivate dealing with this special case.
        raise NotImplementedError("C must have at least two elements")
    elif not all(np.isfinite(C)):
        raise ValueError("C contains non-finite elements")
    elif not all(np.isreal(C)):
        raise ValueError("C contains elements that aren't reals")

    if len(D) <= 1:
        raise NotImplementedError("D must have at least two elements")
    elif D.shape != C.shape:
        raise ValueError("C and D have incompatible shapes")
    elif not all(np.isfinite(D)):
        raise ValueError("D contains non-finite elements")
    elif not all(np.isreal(D)):
        raise ValueError("D contains elements that aren't reals")

    # This definition assume that C and D are variables in the local scope of
    # this function.  In this case, they're the local scope copies of the
    # function's arguments.
    def hfun(z, H0=None):
        eqtol = 1e-8

        p = len(C)

        # Error check under the assumption that it is essentially only MSP,
        # which is under our control, calling this function.
        assert isinstance(z, np.ndarray)
        assert z.ndim == 1
        if len(z) != p:
            raise ValueError("z size incompatible with C & D")
        elif not all(np.isfinite(z)):
            raise ValueError("z contains non-finite elements")
        elif not all(np.isreal(z)):
            raise ValueError("z contains elements that aren't reals")

        if H0 is None:
            h = np.sum(np.abs(D - np.maximum(z, C)))
            g = [[] for _ in range(p)]
            H = [[] for _ in range(p)]

            for i in range(p):
                if z[i] <= C[i] or abs(z[i] - C[i]) < eqtol * max(abs(z[i]), abs(C[i])) or abs(z[i] - C[i]) < eqtol:
                    if C[i] >= D[i]:
                        g[i].append(0)
                        H[i].append("2")
                    if C[i] <= D[i]:
                        g[i].append(0)
                        H[i].append("4")
                if z[i] >= C[i] or abs(z[i] - C[i]) < eqtol * max(abs(z[i]), abs(C[i])) or abs(z[i] - C[i]) < eqtol:
                    if (max(z[i], C[i]) == D[i]) or (abs(max(z[i], C[i]) - D[i]) < eqtol * max(abs(max(z[i], C[i])), abs(D[i]))) or (abs(max(z[i], C[i]) - D[i]) < eqtol):
                        g[i].append(1)
                        g[i].append(-1)
                        H[i].append("1")
                        H[i].append("3")
                    else:
                        g[i].append(np.sign(z[i] - D[i]))
                        if D[i] >= z[i]:
                            H[i].append("3")
                        else:
                            H[i].append("1")

            grads = np.array(list(product(*g))).T

            Hash = ["".join(t) for t in product(*H)]

            return h, grads, Hash
        else:
            K = len(H0)

            h = np.zeros(K)
            grads = np.zeros((p, K))
            vals = np.zeros((p, K))

            for k in range(K):
                for j in range(p):
                    if H0[k][j] == "1":
                        vals[j, k] = -(D[j] - z[j])
                        grads[j, k] = 1
                    elif H0[k][j] == "2":
                        vals[j, k] = -(D[j] - C[j])
                        grads[j, k] = 0
                    elif H0[k][j] == "3":
                        vals[j, k] = D[j] - z[j]
                        grads[j, k] = -1
                    elif H0[k][j] == "4":
                        vals[j, k] = D[j] - C[j]
                        grads[j, k] = 0
                h[k] = np.sum(vals[:, k])

            return h, grads

    # The value of the given actual C and D arguments are fixed within the
    # returned functions only at this point.
    return hfun
