from itertools import product

import numpy as np


def _activities_and_inds(h, z, n=None, atol=1e-8, rtol=1e-8):
    if n is None:
        n = len(z)

    inds = np.where(np.abs(h - z) <= atol + rtol * np.abs(z))[0]

    grads = np.zeros((n, len(inds)))
    Hashes = [str(ind) for ind in inds]

    return inds, grads, Hashes


def censored_L1_loss(z, H0=None, **kwargs):
    """
    This is a generalized version of Womersley's censored L1 loss function.
    """

    C = kwargs["C"]
    D = kwargs["D"]

    eqtol = 1e-8

    # Ensure column vector and collect dimensions
    z = np.asarray(z).flatten()
    C = np.asarray(C).flatten()
    D = np.asarray(D).flatten()
    p = len(C)

    if H0 is None:
        h = np.sum(np.abs(D - np.maximum(z, C)))
        g = [[] for _ in range(p)]
        H = [[] for _ in range(p)]

        for i in range(p):
            if z[i] <= C[i] or abs(z[i] - C[i]) < eqtol * max(abs(z[i]), abs(C[i])) or abs(z[i] - C[i]) < eqtol:
                if C[i] >= D[i]:
                    g[i].append(0)
                    H[i].append("2")
                if C[i] <= D[i]:
                    g[i].append(0)
                    H[i].append("4")
            if z[i] >= C[i] or abs(z[i] - C[i]) < eqtol * max(abs(z[i]), abs(C[i])) or abs(z[i] - C[i]) < eqtol:
                if (max(z[i], C[i]) == D[i]) or (abs(max(z[i], C[i]) - D[i]) < eqtol * max(abs(max(z[i], C[i])), abs(D[i]))) or (abs(max(z[i], C[i]) - D[i]) < eqtol):
                    g[i].append(1)
                    g[i].append(-1)
                    H[i].append("1")
                    H[i].append("3")
                else:
                    g[i].append(np.sign(z[i] - D[i]))
                    if D[i] >= z[i]:
                        H[i].append("3")
                    else:
                        H[i].append("1")

        grads = np.array(list(product(*g))).T

        Hash = ["".join(t) for t in product(*H)]

        return h, grads, Hash
    else:
        K = len(H0)

        h = np.zeros(K)
        grads = np.zeros((p, K))
        vals = np.zeros((p, K))

        for k in range(K):
            for j in range(p):
                if H0[k][j] == "1":
                    vals[j, k] = -(D[j] - z[j])
                    grads[j, k] = 1
                elif H0[k][j] == "2":
                    vals[j, k] = -(D[j] - C[j])
                    grads[j, k] = 0
                elif H0[k][j] == "3":
                    vals[j, k] = D[j] - z[j]
                    grads[j, k] = -1
                elif H0[k][j] == "4":
                    vals[j, k] = D[j] - C[j]
                    grads[j, k] = 0
            h[k] = np.sum(vals[:, k])

        return h, grads


def one_norm(z, H0=None):
    # Evaluates
    #   sum(abs(z_j))

    # Inputs:
    #  z:              [1 x p]   point where we are evaluating h
    #  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate z

    # Outputs:
    #  h: [dbl]                       function value
    #  grads: [p x l]                 gradients of each of the l manifolds active at z
    #  Hash: [1 x l cell of strings]  set of hashes for each of the l manifolds active at z (in the same order as the elements of grads)

    if H0 is None:
        h = np.sum(np.abs(z))

        tol = 1e-8

        grad_lists = [None] * len(z)
        Hash_lists = [None] * len(z)

        for i, element in enumerate(z):
            if element < -tol:
                grad_lists[i] = [-1]
                Hash_lists[i] = ["-"]
            elif element > tol:
                grad_lists[i] = [1]
                Hash_lists[i] = ["+"]
            else:
                # Technically, we should return [1,-1] for the grad entry and
                # ['-','+'] for the Hash, but that causes issues for large dim(z)
                # because we get 2^dim(z) grads.... but they don't matter
                # really for the convex hull calculation
                grad_lists[i] = [0]
                Hash_lists[i] = ["0"]

        all_grad_perms = product(*grad_lists)

        grads = np.array(list(all_grad_perms)).T
        Hash = ["".join(t) for t in product(*Hash_lists)]

        return h, grads, Hash

    else:
        J = len(H0)
        h = np.zeros(J)
        grads = np.ones((len(z), J))

        for k in range(J):
            ztemp = np.copy(z)
            for j in range(len(z)):
                if H0[k][j] == "-":
                    grads[j, k] = -1
                    ztemp[j] *= -1
                elif H0[k][j] == "0":
                    grads[j, k] = 0
                    if ztemp[j] < 0:
                        ztemp[j] *= -1

            h[k] = np.sum(ztemp)

        return h, grads


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
        i1 = np.argmax(z2)
        h = z[i1] ** 2

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
    cs = np.squeeze(kwargs["cs"])

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


def max_gamma_over_KY(z, H0=None):
    """
    Computes h = max_j { z_j / KY_j }, where each z_j represents the output from
    the application-specific function gamma(kappa, Delta, zeta, KY_j).

    Notes
    -----
    - The symbols kappa, Delta, and zeta are domain parameters from
      the physics/application model. They are *not* related to parameters
      in the manifold sampling algorithm itself.
    - The role of hfun in manifold sampling is simply to wrap this
      application objective in the required (value, gradients, hashes)
      interface. The actual Ffun corresponds to gamma(·).

    Inputs
    ------
    z : array-like, shape (11,)
        Values of gamma(...) evaluated at the 11 KY points.
    H0 : optional list of str
        Hashes (indices as strings) of manifolds to evaluate specifically.
        If None, returns active/near-active manifolds at z.
    KY : optional array-like, shape (11,)
        The KY grid; defaults to [0.10, 0.15, ..., 0.60].

    Outputs (H0 is None)
    --------------------
    h : float
        The maximum value over j of z_j / KY_j.
    grads : ndarray, shape (11, l)
        Columns are gradients of each active manifold (dh/dz_j = 1/KY_j).
    Hash : list[str], length l
        String indices of active/near-active manifolds, matching grads columns.

    Outputs (H0 provided)
    ---------------------
    h : ndarray, shape (l,)
        Values z_j / KY_j for requested manifolds.
    grads : ndarray, shape (11, l)
        Gradient columns for requested manifolds.
    """

    KY = np.linspace(0.10, 0.60, 11)  # Fixed KY grid: [0.10, 0.15, ..., 0.60] (11 values)

    z = np.asarray(z, dtype=float).ravel()
    assert z.size == 11, f"Expected 11 values in z (got {z.size})."
    assert KY.size == 11, f"Expected 11 KY values (got {KY.size})."

    # Per-manifold values and gradient magnitudes
    vals = z / KY  # shape (11,)
    grad_mag = 1.0 / KY  # dh/dz_j for active j

    if H0 is None:
        h = float(np.max(vals))
        inds, grads, Hash = _activities_and_inds(h, vals, n=len(z))
        for j in range(len(inds)):
            grads[inds[j], j] = grad_mag[inds[j]]
        return h, grads, Hash
    else:
        H0 = list(H0)
        J = len(H0)
        h = np.zeros(J, dtype=float)
        grads = np.zeros((len(z), J), dtype=float)
        for k in range(J):
            j = int(H0[k])
            h[k] = vals[j]
            grads[j, k] = grad_mag[j]
        return h, grads
