import numpy as np


def pw_minimum_squared(z, H0=None):
    # Evaluates the pointwise minimum function
    #   min_j { z_j^2 }

    # Inputs:
    #  z:              [1 x p]   point where we are evaluating h
    #  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate z

    # Outputs:
    #  h: [dbl]                       function value
    #  grads: [p x l]                 gradients of each of the l manifolds active at z
    #  Hash: [1 x l cell of strings]  set of hashes for each of the l manifolds active at z (in the same order as the elements of grads)

    n = len(z)

    if H0 is None:
        z2 = z ** 2
        h = np.min(z2)

        atol = 1e-8
        rtol = 1e-8
        inds = np.where(np.abs(h - z2) <= atol + rtol * np.abs(z2))[0]

        grads = np.zeros((n, len(inds)))

        Hash = [str(ind) for ind in inds]
        for j in range(len(inds)):
            grads[inds[j], j] = 2 * z[inds[j]]

        return h, grads, Hash

    else:
        J = len(H0)
        h = np.zeros(J)
        grads = np.zeros((n, J))

        for k in range(J):
            j = int(H0[k])
            h[k] = z[j] ** 2
            grads[j, k] = 2 * z[j]

        return h, grads
