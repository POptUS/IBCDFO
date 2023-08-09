import numpy as np


def pw_minimum(z, H0):
    # Evaluates the pointwise maximum function
    #   min_j { z_j }

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
        h = np.amin(z)
        atol = 1e-08
        rtol = 1e-08
        inds = find(np.abs(h - z) <= atol + rtol * np.abs(z))
        grads = np.zeros((n, len(inds)))
        Hash = cell(1, len(inds))
        for j in np.arange(1, len(inds) + 1).reshape(-1):
            Hash[j] = int2str(inds(j))
            grads[inds[j], j] = 1
    else:
        if len(varargin) == 2:
            J = len(H0)
            h = np.zeros((1, J))
            grads = np.zeros((len(z), J))
            for k in np.arange(1, J + 1).reshape(-1):
                j = str2num(H0[k])
                h[k] = z(j)
                grads[j, k] = 1
        else:
            raise Exception("Too many inputs to function")

    return h, grads, Hash
