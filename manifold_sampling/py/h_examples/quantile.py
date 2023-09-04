import numpy as np


def quantile(z, H0=None):
    # Evaluates the q^th quantile of the values
    #  { z_j^2 }

    # Inputs:
    #  z:              [1 x p]   point where we are evaluating h
    #  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate

    # Outputs:
    #  h: [dbl]                       function value
    #  grads: [p x l]                 gradients of each of the l quadratics active at z
    #  Hash: [1 x l cell of strings]  set of hashes for each of the l quadratics active at z (in the same order as the elements of grads)

    q = 1

    n = len(z)
    z2 = z**2
    sorted_inds = np.argsort(z2)
    z2_sort = z2[sorted_inds]
    z_sort = z[sorted_inds]

    if H0 is None:
        h = z2_sort[q]
        atol = 1e-08
        rtol = 1e-08
        inds = np.where(np.abs(h - z2_sort) <= atol + rtol * np.abs(z2_sort))[0]

        grads = np.zeros((n, len(inds)))
        Hash = [str(ind) for ind in inds]

        for j in range(len(inds)):
            grads[inds[j], j] = 2 * z_sort[inds[j]]

        return h, grads, Hash
    else:
        J = len(H0)
        h = np.zeros(J)
        grads = np.zeros((len(z), J))

        for k in range(J):
            j = int(H0[k])
            h[k] = z2_sort[j]
            grads[j, k] = 2 * z_sort[j]

        return h, grads
