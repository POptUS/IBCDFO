import numpy as np

from .._bound_checks import snap_to_bounds


def call_user_scripts(nf, X, F, hF, Hash, Ffun, hfun, x_in, tol, Low, Upp, allow_recalc=0):
    if len(x_in) != len(Upp) or len(x_in) != len(Low):
        raise ValueError("Input vector dimensions do not match the bounds.")

    if allow_recalc is None:
        allow_recalc = 0

    xnew = snap_to_bounds(x_in, Low, Upp, tol["snap_tol"], "The requested point")

    for i in range(len(xnew)):
        if Upp[i] - xnew[i] < np.finfo(float).eps * np.abs(Upp[i]) and Upp[i] > xnew[i]:
            xnew[i] = Upp[i]
            print("eps project!")
        elif xnew[i] - Low[i] < np.finfo(float).eps * np.abs(Low[i]) and Low[i] < xnew[i]:
            xnew[i] = Low[i]
            print("eps project!")

    xnew_hashable = tuple(xnew)

    if not allow_recalc and xnew_hashable in [tuple(row) for row in X[:nf, :]]:
        raise ValueError("Your method requested a point that has already been requested. FAIL!")

    nf += 1
    X[nf] = xnew
    F[nf] = Ffun(X[nf])
    hF[nf], _, hashes_at_nf = hfun(F[nf])
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

    return nf, X, F, hF, Hash, hashes_at_nf
