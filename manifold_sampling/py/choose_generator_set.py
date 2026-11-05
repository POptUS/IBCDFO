import numpy as np
from scipy.spatial.distance import cdist


def choose_generator_set(X, Hash, xk_in, nf, delta, F, hfun):
    r"""
    Choose the generator set ``D_k`` for the manifold sampling subproblem at the
    current trust region center ``X[xk_in]``. The generators are the gradients, at
    the center ``F[xk_in]``, of a set of selection functions (manifolds) of h.

    The selection functions are those active at the center, plus those active at
    evaluated points near the center. Each is evaluated at the center
    ``F[xk_in]``, not at the point where it was found to be active. For an
    evaluated point ``X[i]``, let ``h_i`` be the value at ``F[xk_in]`` of the first
    selection function active at ``X[i]``, and let ``hF_at_xk_in`` be the
    corresponding value for the center (that is, h at the center). All selection
    functions active at ``X[i]`` are added when
    ``||X[i] - X[xk_in]||_inf <= delta`` and either
    ``h_i <= hF_at_xk_in`` or ``||X[i] - X[xk_in]||_inf <= min(1, delta)**2``,
    where each distance test has a relative tolerance of 1e-8. Selection
    functions whose gradients coincide are kept only once.

    :param X: Points evaluated (rows ``0`` through ``nf`` are in use).
    :param Hash: ``Hash[i]`` holds the hashes of the selection functions active at
        ``X[i]``.
    :param xk_in: Zero-based index of the trust region center.
    :param nf: Zero-based index of the most recently evaluated point.
    :param delta: Trust region radius.
    :param F: Function values at the points in ``X``.
    :param hfun: Function implementing h; given ``z`` and a set of hashes, returns
        the values and gradients of those selection functions.

    :return: ``(D_k, Act_Z_k, f_k)`` where ``D_k`` is the generator set, a
        :math:`(p, l)` array whose columns are the gradients, at ``F[xk_in]``, of
        the ``l`` chosen selection functions, ``Act_Z_k`` holds their hashes (in the
        order of the columns of ``D_k``), and ``f_k`` holds their values at
        ``F[xk_in]``.
    """
    Act_Z_k = Hash[xk_in]

    hF_at_xk_in, _ = hfun(F[xk_in, :], Act_Z_k)
    XkDist = cdist(X[: nf + 1], X[xk_in : xk_in + 1], metric="chebyshev")
    delta1 = delta * (1 + 1e-8)
    delta2 = min(1, delta) ** 2 * (1 + 1e-8)

    for i, XkDi in enumerate(XkDist):
        if XkDi <= delta1:
            Act_tmp = Hash[i]
            h_i, _ = hfun(F[xk_in], Act_tmp)
            if h_i[0] <= hF_at_xk_in[0] or XkDi <= delta2:
                if i != xk_in:
                    Act_Z_k = np.unique(np.concatenate((Act_Z_k, Act_tmp)), axis=0)

    Act_Z_k = np.asarray(Act_Z_k)
    f_k, D_k = hfun(F[xk_in], Act_Z_k)
    unique_indices = np.unique(D_k, axis=1, return_index=True)[1]
    D_k = D_k[:, unique_indices]
    Act_Z_k = Act_Z_k[unique_indices]
    f_k = f_k[unique_indices]

    return D_k, Act_Z_k, f_k
