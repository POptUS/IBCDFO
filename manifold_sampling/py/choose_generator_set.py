import numpy as np
from scipy.spatial.distance import cdist


def _unique_axis1_indices_preserve_shape(A):
    """
    Faster replacement for np.unique(A, axis=1, return_index=True)[1] when safe.
    Falls back to NumPy otherwise.
    """
    A = np.asarray(A)

    if A.ndim == 2 and A.dtype != object:
        AT = np.ascontiguousarray(A.T)
        col_view = AT.view(np.dtype((np.void, AT.dtype.itemsize * AT.shape[1])))
        _, idx = np.unique(col_view, return_index=True)
        idx.sort()  # preserve first-occurrence order
        return idx

    return np.unique(A, axis=1, return_index=True)[1]


def choose_generator_set(X, Hash, xkin, nf, delta, F, hfun):
    Act_Z_k = Hash[xkin]

    hxkin, _ = hfun(F[xkin, :], Act_Z_k)
    XkDist = cdist(X[: nf + 1], X[xkin : xkin + 1], metric="chebyshev")
    delta1 = delta * (1 + 1e-8)
    delta2 = min(1, delta) ** 2 * (1 + 1e-8)

    act_blocks = [Act_Z_k] # Build this and just do one call to unique
    for i, XkDi in enumerate(XkDist):
        if XkDi <= delta1:
            Act_tmp = Hash[i]
            h_i, _ = hfun(F[xkin], Act_tmp)
            if h_i[0] <= hxkin[0] or XkDi <= delta2:
                if i != xkin:
                    act_blocks.append(Act_tmp)

    if len(act_blocks) > 1:
        Act_Z_k = np.unique(np.concatenate(act_blocks))
    else:
        Act_Z_k = np.asarray(Act_Z_k)
    f_k, D_k = hfun(F[xkin], Act_Z_k)
    unique_indices = _unique_axis1_indices_preserve_shape(D_k)
    D_k = D_k[:, unique_indices]
    Act_Z_k = Act_Z_k[unique_indices]
    f_k = f_k[unique_indices]

    return D_k, Act_Z_k, f_k
