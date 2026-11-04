import numpy as np
from scipy.spatial.distance import cdist


def choose_generator_set(X, Hash, xk_in, nf, delta, F, hfun):
    Act_Z_k = Hash[xk_in]

    hxkin, _ = hfun(F[xk_in, :], Act_Z_k)
    XkDist = cdist(X[: nf + 1], X[xk_in : xk_in + 1], metric="chebyshev")
    delta1 = delta * (1 + 1e-8)
    delta2 = min(1, delta) ** 2 * (1 + 1e-8)

    for i, XkDi in enumerate(XkDist):
        if XkDi <= delta1:
            Act_tmp = Hash[i]
            h_i, _ = hfun(F[xk_in], Act_tmp)
            if h_i[0] <= hxkin[0] or XkDi <= delta2:
                if i != xk_in:
                    Act_Z_k = np.unique(np.concatenate((Act_Z_k, Act_tmp)), axis=0)

    Act_Z_k = np.asarray(Act_Z_k)
    f_k, D_k = hfun(F[xk_in], Act_Z_k)
    unique_indices = np.unique(D_k, axis=1, return_index=True)[1]
    D_k = D_k[:, unique_indices]
    Act_Z_k = Act_Z_k[unique_indices]
    f_k = f_k[unique_indices]

    return D_k, Act_Z_k, f_k
