import numpy as np


def piecewise_quadratic(z, H0):
    # Evaluates the piecewise quadratic function
    #   max_j { || z - z_j ||_{Q_j}^2 + b_j }

    # Inputs:
    #  z:              [1 x p]   point where we are evaluating h
    #  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate

    # Outputs:
    #  h: [dbl]                       function value
    #  grads: [p x l]                 gradients of each of the l quadratics active at z
    #  Hash: [1 x l cell of strings]  set of hashes for each of the l quadratics active at z (in the same order as the elements of grads)

    # Hashes are output (and must be input) in the following fashion:
    #   Hash{i} = 'j' if quadratic j is active at z (or H0{i} = 'j' if the
    #   value/gradient of quadratic j at z is desired)

    global Qs, zs, cs, h_activity_tol
    if len(h_activity_tol) == 0:
        h_activity_tol = 0

    z = z
    if len(varargin) == 1:
        n, J = zs.shape
        manifolds = np.zeros((1, J))
        for j in np.arange(1, J + 1).reshape(-1):
            manifolds[j] = np.transpose((z - zs[:, j])) * Qs[:, :, j] * (z - zs[:, j]) + cs(j)
        h = np.amax(manifolds)
        atol = h_activity_tol
        rtol = h_activity_tol
        inds = find(np.abs(h - manifolds) <= atol + rtol * np.abs(manifolds))
        grads = np.zeros((n, len(inds)))
        Hash = cell(1, len(inds))
        for j in np.arange(1, len(inds) + 1).reshape(-1):
            Hash[j] = int2str(inds[j])
            grads[:, j] = 2 * Qs[:, :, inds[j]] * (z - zs[:, inds[j]])
    else:
        if len(varargin) == 2:
            J = len(H0)
            h = np.zeros((1, J))
            grads = np.zeros((len(z), J))
            for k in np.arange(1, J + 1).reshape(-1):
                j = str2num(H0[k])
                h[k] = np.transpose((z - zs[:, j])) * Qs[:, :, j] * (z - zs[:, j]) + cs[j]
                grads[:, k] = 2 * Qs[:, :, j] * (z - zs[:, j])
        else:
            raise Exception("Too many inputs to function")

    return h, grads, Hash
