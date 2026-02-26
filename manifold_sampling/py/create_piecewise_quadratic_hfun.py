import numpy as np

from .general_nonsmooth_h_funs import _activities_and_inds


def create_piecewise_quadratic_hfun(Qs, zs, cs):
    r"""
    Create an :math:`\hfun` function using the given :math:`Q_j, \zvec_j, c_j`
    parameter values for constructing and using the manifold sampling piecewise
    quadratic objective function

    .. math::

        f(\psp; Q_1, \cdots, Q_l, \zvec_1, \cdots, \zvec_l, c_1, \cdots, c_l)
            & = \hfun\left(\zvec(\psp); Q_1, \cdots, Q_l, \zvec_1, \cdots, \zvec_l, c_1, \cdots, c_l\right)\\
            & = \max_{j\in\set{1, \cdots, l}}\set{\norm{\zvec(\psp) - \zvec_j}^2_{Q_j} + c_j}.

    This family of :math:`h` functions is included in the package as an example
    and might not be useful for constructing practical optimization problems.

    .. todo::

        * Please check if the above formula is correct.

    :param Qs: :math:`m \times m \times l` numpy array that contains the
        :math:`Q_j \in \R^{m \times m}` parameter values.  Note that at least
        one :math:`Q_j` should be symmetric positive definite in order for the
        associated optimization problem to be formally well-defined.  There is
        no error checking to confirm that given ``Qs`` arguments satisfy this
        requirement.
    :param zs: :math:`m \times l` numpy array that contains the :math:`\zvec_j
        \in \R^m` parameter values
    :param cs: 1D numpy array of length :math:`l` that specifies the :math:`c_j`
        parameter values
    :return: ``hfun`` constructed with the given :math:`Q_j, \zvec_j, c_j` that
        is compatible only with :math:`\zvec` arguments whose lengths are
        compatible with the shapes of ``Qs`` and ``zs``.
    """
    # ----- CREATE LOCAL-SCOPE COPIES
    # This prevents the code here and in hfun from accidentally altering the
    # contents of the actual arrays passed into the function.
    #
    # IMPORTANT: Aside from making copies don't alter Qs, zs, cs anywhere in
    # this function.
    if not isinstance(Qs, np.ndarray):
        raise TypeError("Qs is not a numpy array")
    Qs = np.atleast_3d(np.squeeze(Qs.copy()))

    if not isinstance(zs, np.ndarray):
        raise TypeError("zs is not a numpy array")
    zs = np.atleast_2d(np.squeeze(zs.copy()))

    if not isinstance(cs, np.ndarray):
        raise TypeError("cs is not a numpy array")
    cs = np.atleast_1d(np.squeeze(cs.copy()))

    # ----- ERROR CHECK ARGUMENTS
    if Qs.ndim != 3:
        raise ValueError("Qs is not a 3D numpy array")
    elif Qs.shape[0] != Qs.shape[1]:
        raise ValueError("Qs must be 3D numpy array with the first two dimensions equal")
    elif not all(np.isfinite(Qs.flatten())):
        raise ValueError("Qs contains non-finite elements")
    elif not all(np.isreal(Qs.flatten())):
        raise ValueError("Qs contains elements that aren't reals")

    if zs.ndim != 2:
        raise ValueError("zs is not a 2D numpy array")
    elif (zs.shape[0] != Qs.shape[0]) or (zs.shape[1] != Qs.shape[2]):
        raise ValueError("Qs and zs have incompatible shapes")
    elif not all(np.isfinite(zs.flatten())):
        raise ValueError("zs contains non-finite elements")
    elif not all(np.isreal(zs.flatten())):
        raise ValueError("zs contains elements that aren't reals")

    if cs.ndim != 1:
        raise ValueError("cs is not a 1D numpy array")
    elif len(cs) != Qs.shape[2]:
        raise ValueError("Qs and cs have incompatible shapes")
    elif not all(np.isfinite(cs)):
        raise ValueError("cs contains non-finite elements")
    elif not all(np.isreal(cs)):
        raise ValueError("cs contains elements that aren't reals")

    # This definition assume that Qs, zs, and cs are variables in the local
    # scope of this function.  In this case, they're the local scope copies of
    # the function's arguments.
    def hfun(z, H0=None):
        # Inputs:
        #  z:              [1 x p]   point where we are evaluating h
        #  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate
        #
        # Outputs:
        #  h: [dbl]                       function value
        #  grads: [p x l]                 gradients of each of the l quadratics active at z
        #  Hash: [1 x l cell of strings]  set of hashes for each of the l quadratics active at z (in the same order as the elements of grads)
        #
        # Hashes are output (and must be input) in the following fashion:
        #   Hash{i} = 'j' if quadratic j is active at z (or H0{i} = 'j' if the
        #   value/gradient of quadratic j at z is desired)

        # Error check under the assumption that it is essentially only MSP,
        # which is under our control, calling this function.  However, assume
        # that MSP is *not* confirming that z values are finite reals.
        assert isinstance(z, np.ndarray)
        assert z.ndim == 1
        if len(z) != zs.shape[0]:
            raise ValueError("z size incompatible with zs")
        elif not all(np.isfinite(z)):
            raise ValueError("z contains non-finite elements")
        elif not all(np.isreal(z)):
            raise ValueError("z contains elements that aren't reals")

        if H0 is None:
            n, J = zs.shape
            manifolds = np.zeros(J)
            for j in range(J):
                manifolds[j] = np.dot(np.dot((z - zs[:, j]), Qs[:, :, j]), (z - zs[:, j])) + cs[j]

            h = np.max(manifolds)

            inds, grads, Hash = _activities_and_inds(h, manifolds, n=n)

            for j in range(len(inds)):
                grads[:, j] = 2 * np.dot(Qs[:, :, inds[j]], (z - zs[:, inds[j]]))

            return h, grads, Hash

        else:
            J = len(H0)
            h = np.zeros(J)
            grads = np.zeros((len(z), J))

            for k in range(J):
                j = int(H0[k])
                h[k] = np.dot(np.dot((z - zs[:, j]), Qs[:, :, j]), (z - zs[:, j])) + cs[j]
                grads[:, k] = 2 * np.dot(Qs[:, :, j], (z - zs[:, j]))

            return h, grads

    # The value of the given actual Qs, zs, and cs arguments are fixed within
    # the returned functions only at this point.
    return hfun
