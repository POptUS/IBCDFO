import numpy as np


def choose_generator_set(X, Hash, gentype, xkin, nf, delta, F, hfun):
    Act_Z_k = Hash[xkin]

    if gentype == 2:
        for i in [ind for ind in range(nf) if ind != xkin]:
            if np.linalg.norm(X[xkin] - X[i]) <= delta * (1 + 1e-8):
                Act_tmp = Hash[i]
                Act_Z_k = np.concatenate((Act_Z_k, Act_tmp))

    elif gentype == 3:
        hxkin, _ = hfun(F[xkin, :], Act_Z_k)
        for i in [ind for ind in range(nf) if ind != xkin]:
            Act_tmp = Hash[i]
            h_i, _ = hfun(F[xkin], Act_tmp)
            if np.linalg.norm(X[xkin] - X[i], ord=np.inf) <= delta * (1 + 1e-8) and h_i[0] <= hxkin[0]:
                Act_Z_k = np.concatenate((Act_Z_k, Act_tmp))
            elif np.linalg.norm(X[xkin] - X[i], ord=np.inf) <= delta**2 * (1 + 1e-8) and h_i[0] > hxkin[0]:
                Act_Z_k = np.concatenate((Act_Z_k, Act_tmp))

    Act_Z_k = np.unique(Act_Z_k)
    f_k, D_k = hfun(F[xkin], Act_Z_k)
    unique_indices = np.unique(D_k, axis=1, return_index=True)[1]
    D_k = D_k[:, unique_indices]
    Act_Z_k = Act_Z_k[unique_indices]
    f_k = f_k[unique_indices]

    return D_k, Act_Z_k, f_k
