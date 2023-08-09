import numpy as np


def choose_generator_set(X, Hash, gentype, xkin, nf, delta, F, hfun):
    # Returns:

    # D_k:     A set of gradients of h_j at points near X(xkin,:)
    # Act_Z_k: A set of hashes for points in D_k (may contain duplicates in the
    #          gentype == 2 case if D_k has multiple gradients from a given hash)

    Act_Z_k = Hash[xkin, not cellfun(isempty, Hash[xkin, :])]
    if gentype == 2:
        for i in np.array([np.arange(1, xkin - 1 + 1), np.arange(xkin + 1, nf + 1)]).reshape(-1):
            if np.linalg.norm(X[xkin] - X[i]) <= delta * (1 + 1e-08):
                Act_tmp = Hash[i, not cellfun(isempty, Hash[i])]
                Act_Z_k = np.array([Act_Z_k, Act_tmp])
    else:
        if gentype == 3:
            hxkin = hfun(F[xkin, :], Act_Z_k)
            for i in np.array([np.arange(1, xkin - 1 + 1), np.arange(xkin + 1, nf + 1)]).reshape(-1):
                Act_tmp = Hash[i, not cellfun(isempty, Hash[i])]
                h_i = hfun(F[xkin], Act_tmp)
                if np.linalg.norm(X[xkin] - X[i], "inf") <= delta * (1 + 1e-08) and h_i[1] <= hxkin[1]:
                    Act_Z_k = np.array([Act_Z_k, Act_tmp])
                else:
                    if np.linalg.norm(X[xkin] - X[i], "inf") <= delta**2 * (1 + 1e-08) and h_i[1] > hxkin[1]:
                        Act_Z_k = np.array([Act_Z_k, Act_tmp])

    f_k, D_k = hfun(F[xkin, :], Act_Z_k)
    D_k, inds = unique(np.transpose(D_k), "rows")
    D_k = np.transpose(D_k)
    Act_Z_k = Act_Z_k[inds]
    f_k = f_k[inds]
    return D_k, Act_Z_k, f_k
