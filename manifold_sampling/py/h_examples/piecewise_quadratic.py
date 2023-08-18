import numpy as np


def piecewise_quadratic(z, H0=None, **kwargs):
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

    Qs = kwargs["Qs"]
    zs = kwargs["zs"]
    cs = kwargs["cs"]

    if H0 is None:
        n, J = zs.shape
        manifolds = np.zeros(J)
        for j in range(J):
            manifolds[j] = np.dot(np.dot((z - zs[:, j]), Qs[:, :, j]), (z - zs[:, j])) + cs[j]

        h = np.max(manifolds)

        atol = 1e-8
        rtol = 1e-8
        inds = np.where(np.abs(h - manifolds) <= atol + rtol * np.abs(manifolds))[0]

        grads = np.zeros((n, len(inds)))

        Hash = [str(ind) for ind in inds]
        for j in range(len(inds)):
            grads[:, j] = 2 * np.dot(Qs[:, :, inds[j]], (z - zs[:, inds[j]]))

        return h, grads, Hash

    else:
        J = len(H0)
        h = np.zeros(J)
        grads = np.zeros((len(z), J))

        for k in range(J):
            j = int(H0[k])
            h[k] = np.dot(np.dot((z - zs[:, j]), Qs[:, :, j]), (z - zs[:, j])) + cs[j]
            grads[:, k] = 2 * np.dot(Qs[:, :, j], (z - zs[:, j]))

        return h, grads
