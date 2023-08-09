import numpy as np


def max_sum_beta_plus_const_viol(z, H0):
    # This the outer h function required by manifold sampling.
    # If z \in R^p
    # It encodes the objective
    #    max_{i = 1,...,p1} z_i + alpha *sum_{i = p1+1}^{p} max(z_i, 0)^2

    # Hashes are output (and must be input) in the following fashion:
    #   Hash elements are strings of p integers.
    #     0 in position 1 <= i <= p1 means max_{i = 1,...,p1} z_i > z_i
    #     1 in position 1 <= i <= p1 means max_{i = 1,...,p1} z_i = z_i
    #     0 in position p1+1 <= i <= p means max(z_i,0) = 0
    #     1 in position p1+1 <= i <= p means max(z_i,0) = z_i

    #   Similarly, if H0 has a 1 in position i uses z_i in the calculation of h and grads.

    # Get data from outside of this function
    global h_activity_tol, p1, alpha
    p = len(z)
    if len(varargin) == 1:
        h1 = np.amax(z(np.arange(1, p1 + 1)))
        h2 = alpha * sum(np.amax(z(np.arange(p1 + 1, end() + 1)), 0) ** 2)
        h = h1 + h2
        atol = h_activity_tol
        rtol = h_activity_tol
        inds1 = find(np.abs(h1 - z(np.arange(1, p1 + 1))) <= atol + rtol * np.abs(z(np.arange(1, p1 + 1))))
        inds2 = p1 + find(z(np.arange(p1 + 1, end() + 1)) >= -rtol)
        grads = np.zeros((p, len(inds1)))
        Hashes = cell(1, len(inds1))
        for j in np.arange(1, len(inds1) + 1).reshape(-1):
            hash = dec2bin(0, p)
            hash[inds1[j]] = "1"
            hash[inds2] = "1"
            Hashes[j] = hash
            grads[inds1[j], j] = 1
            grads[inds2, j] = alpha * 2 * z(inds2)
    else:
        if len(varargin) == 2:
            J = len(H0)
            h = np.zeros((1, J))
            grads = np.zeros((p, J))
            for k in np.arange(1, J + 1).reshape(-1):
                max_ind = find(H0[k](np.arange(1, p1 + 1)) == "1")
                assert_(len(max_ind) == 1, "I don't know what to do in this case")
                grads[max_ind, k] = 1
                h1 = z(max_ind)
                const_viol_inds = p1 + find(H0[k](np.arange(p1 + 1, end() + 1)) == "1")
                if len(const_viol_inds) == 0:
                    h2 = 0
                else:
                    grads[const_viol_inds, k] = alpha * 2 * z(const_viol_inds)
                    h2 = alpha * sum(np.amax(z(const_viol_inds), 0) ** 2)
                h[k] = h1 + h2
        else:
            raise Exception("Too many inputs to function")

    return h, grads, Hashes
