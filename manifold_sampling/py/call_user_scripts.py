import numpy as np


def call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, x_in, tol, L, U, allow_recalc=0):
    if len(x_in) != len(U) or len(x_in) != len(L):
        raise ValueError("Input vector dimensions do not match the bounds.")

    if allow_recalc is None:
        allow_recalc = 0

    xnew = np.minimum(U, np.maximum(L, x_in))

    for i in range(len(xnew)):
        if U[i] - xnew[i] < np.finfo(float).eps * np.abs(U[i]) and U[i] > xnew[i]:
            xnew[i] = U[i]
            print("eps project!")
        elif xnew[i] - L[i] < np.finfo(float).eps * np.abs(L[i]) and L[i] < xnew[i]:
            xnew[i] = L[i]
            print("eps project!")

    xnew_hashable = tuple(xnew)

    if not allow_recalc and xnew_hashable in [tuple(row) for row in X[:nf, :]]:
        raise ValueError("Your method requested a point that has already been requested. FAIL!")

    nf += 1
    X[nf] = xnew
    F[nf] = Ffun(X[nf])
    h[nf], _, hashes_at_nf = hfun(F[nf])
    Hash[nf] = hashes_at_nf
    if np.any(np.isnan(F[nf, :])):
        raise ValueError("Got a NaN. FAIL!")

    if tol["hfun_test_mode"]:
        assert isinstance(hashes_at_nf, list), "hfun must return a list of hashes"
        # assert all(isinstance(hash_val, str) for hash_val in hashes_at_nf), "Hashes must be strings"

        h_dummy1, grad_dummy1, hash_dummy = hfun(F[nf, :])
        h_dummy2, grad_dummy2 = hfun(F[nf, :], hash_dummy)
        # if not np.any(np.abs(h_dummy1 - h_dummy2) <= 1e-16):
        #     print("DEBUG: ", np.abs(h_dummy1 - h_dummy2), flush=True)
        assert np.any(np.abs(h_dummy1 - h_dummy2) <= 1e-8), "hfun values don't agree when " + hfun.__name__ + " is re-called with the same inputs"
        assert np.all(grad_dummy1 == grad_dummy2), "hfun gradients don't agree when " + hfun.__name__ + " is being re-called with the same inputs"

    return nf, X, F, h, Hash, hashes_at_nf
