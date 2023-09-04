import numpy as np


def pw_minimum(z, H0=None):
    # Evaluates the pointwise minimum function
    #   min_j { z_j }

    # Inputs:
    #  z:              [1 x p]   point where we are evaluating h
    #  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate z

    # Outputs:
    #  h: [dbl]                       function value
    #  grads: [p x l]                 gradients of each of the l manifolds active at z
    #  Hash: [1 x l cell of strings]  set of hashes for each of the l manifolds active at z (in the same order as the elements of grads)

    n = len(z)

    if H0 is None:
        h = np.min(z)

        atol = 1e-8
        rtol = 1e-8
        inds = np.where(np.abs(h - z) <= atol + rtol * np.abs(z))[0]

        grads = np.zeros((n, len(inds)))
        Hash = [str(ind) for ind in inds]

        for j in range(len(inds)):
            grads[inds[j], j] = 1

        return h, grads, Hash

    else:
        J = len(H0)
        h = np.zeros(J)
        grads = np.zeros((len(z), J))

        for k in range(J):
            j = int(H0[k])
            h[k] = z[j]
            grads[j, k] = 1

        return h, grads
