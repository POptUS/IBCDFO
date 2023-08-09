import numpy as np


def pw_maximum_squared(z, H0):
    # Evaluates the pointwise maximum function
    #   max_j { z_j^2 }

    # Inputs:
    #  z:              [1 x p]   point where we are evaluating h
    #  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate z

    # Outputs:
    #  h: [dbl]                       function value
    #  grads: [p x l]                 gradients of each of the l manifolds active at z
    #  Hash: [1 x l cell of strings]  set of hashes for each of the l manifolds active at z (in the same order as the elements of grads)

    z = z
    n = len(z)
    if len(varargin) == 1:
        z2 = z**2
        h = np.amax(z2)
        atol = 1e-08
        rtol = 1e-08
        inds = find(np.abs(h - z2) <= atol + rtol * np.abs(z2))
        grads = np.zeros((n, len(inds)))
        Hash = cell(1, len(inds))
        for j in np.arange(1, len(inds) + 1).reshape(-1):
            Hash[j] = int2str(inds(j))
            grads[inds[j], j] = 2 * z(inds(j))
    else:
        if len(varargin) == 2:
            J = len(H0)
            h = np.zeros((1, J))
            grads = np.zeros((len(z), J))
            for k in np.arange(1, J + 1).reshape(-1):
                j = str2num(H0[k])
                h[k] = z(j) ** 2
                grads[j, k] = 2 * z(j)
        else:
            raise Exception("Too many inputs to function")

    return h, grads, Hash
