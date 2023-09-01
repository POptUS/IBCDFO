import numpy as np
import numpy.linalg as LA
from scipy import spatial


def pw_maximum_eig(z, H0=None):
    n2 = len(z)
    n = np.int(np.sqrt(n2))

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
