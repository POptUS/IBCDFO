import numpy as np
import numpy.linalg as LA
from scipy import spatial


def _activities_and_inds(h, z, n=None, atol=1e-8, rtol=1e-8):
    if n is None:
        n = len(z)

    inds = np.where(np.abs(h - z) <= atol + rtol * np.abs(z))[0]

    grads = np.zeros((n, len(inds)))
    Hashes = [str(ind) for ind in inds]

    return inds, grads, Hashes


def pw_maximum(z, H0=None):
    # Evaluates the pointwise maximum function
    #   max_j { z_j }

    # Inputs:
    #  z:              [1 x p]   point where we are evaluating h
    #  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate z

    # Outputs:
    #  h: [dbl]                       function value
    #  grads: [p x l]                 gradients of each of the l manifolds active at z
    #  Hash: [1 x l cell of strings]  set of hashes for each of the l manifolds active at z (in the same order as the elements of grads)

    if H0 is None:
        h = np.max(z)

        inds, grads, Hash = _activities_and_inds(h, z)

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


def pw_maximum_squared(z, H0=None):
    # Evaluates the pointwise maximum function
    #   max_j { z_j^2 }

    # Inputs:
    #  z:              [1 x p]   point where we are evaluating h
    #  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate z

    # Outputs:
    #  h: [dbl]                       function value
    #  grads: [p x l]                 gradients of each of the l manifolds active at z
    #  Hash: [1 x l cell of strings]  set of hashes for each of the l manifolds active at z (in the same order as the elements of grads)

    if H0 is None:
        z2 = z**2
        h = np.max(z2)

        inds, grads, Hash = _activities_and_inds(h, z2)

        for j in range(len(inds)):
            grads[inds[j], j] = 2 * z[inds[j]]

        return h, grads, Hash

    else:
        J = len(H0)
        h = np.zeros(J)
        grads = np.zeros((len(z), J))

        for k in range(J):
            j = int(H0[k])
            h[k] = z[j] ** 2
            grads[j, k] = 2 * z[j]

        return h, grads


def pw_maximum_eig(z, H0=None):
    n2 = len(z)
    n = int(np.sqrt(n2))

    # reshape z into a matrix
    M = np.reshape(z, (n, n))

    # compute eigendecomposition
    [eigvals, eigvecs] = LA.eig(M)
    sorted_indices = np.argsort(eigvals)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[sorted_indices]

    if H0 is None:
        h = np.max(eigvals)

        atol = 1e-8
        rtol = 1e-8
        inds = np.where(np.abs(h - eigvals) <= atol + rtol * np.abs(eigvals))[0]

        selection_grads = np.zeros((n, len(inds)))
        grads = np.zeros((n2, len(inds)))
        Hash = [[] for i in range(len(inds))]
        for j in range(len(inds)):
            selection_grads[inds[j], j] = 1
            grad = eigvecs @ np.diag(selection_grads[:, j]) @ eigvecs.T
            grads[:, j] = np.reshape(grad, (1, n2))
            # rounding Hash to 1 digit on purpose (this will make sense, see else statement)
            Hash[j] = [str(np.round(val, 1)) for val in eigvecs[inds[j]]]

        return h, grads, Hash

    else:
        len_inds = len(H0)
        h = np.zeros(len_inds)
        grads = np.zeros((n2, len_inds))
        selection_grads = np.zeros((n, len_inds))

        tree = spatial.KDTree(eigvecs)

        H0 = np.array(H0, dtype=float)

        for k in range(len_inds):
            _, ind = tree.query(H0[k])
            h[k] = eigvals[ind]

            selection_grads[ind, k] = 1
            grad = eigvecs @ np.diag(selection_grads[:, k]) @ eigvecs.T
            grads[:, k] = np.reshape(grad, (1, n2))

        return h, grads


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

        inds, grads, Hash = _activities_and_inds(h, manifolds, n=n)

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

    if H0 is None:
        h = np.min(z)

        inds, grads, Hash = _activities_and_inds(h, z)

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

    if H0 is None:
        z2 = z**2
        h = np.min(z2)

        inds, grads, Hash = _activities_and_inds(h, z2)

        for j in range(len(inds)):
            grads[inds[j], j] = 2 * z[inds[j]]

        return h, grads, Hash

    else:
        J = len(H0)
        h = np.zeros(J)
        grads = np.zeros((len(z), J))

        for k in range(J):
            j = int(H0[k])
            h[k] = z[j] ** 2
            grads[j, k] = 2 * z[j]

        return h, grads


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

    z2 = z**2
    sorted_inds = np.argsort(z2)
    z2_sort = z2[sorted_inds]
    z_sort = z[sorted_inds]

    if H0 is None:
        h = z2_sort[q]
        inds, grads, Hash = _activities_and_inds(h, z2_sort)

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
