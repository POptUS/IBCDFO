import numpy as np


def call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, x_in, tol, L, U, allow_recalc):
    # Call user scripts Ffun and hfun; saves their output to the appropriate arrays
    if len(varargin) < 12:
        allow_recalc = 0

    xnew = np.amin(U, np.amax(L, x_in))
    for i in np.arange(1, len(xnew) + 1).reshape(-1):
        if U[i] - xnew[i] < eps * np.abs(U[i]) and U[i] > xnew[i]:
            xnew[i] = U[i]
            print("eps project!")
        else:
            if xnew[i] - L[i] < eps * np.abs(L[i]) and L[i] < xnew[i]:
                xnew[i] = L[i]
                print("eps project!")

    if not allow_recalc and ismember(xnew, X[np.arange(1, nf + 1), :], "rows"):
        raise Exception("Your method requested a point that has already been requested. FAIL!")

    nf = nf + 1
    X[nf, :] = xnew
    F[nf, :] = Ffun(X[nf, :])
    h[nf], __, hashes_at_nf = hfun(F[nf, :])
    Hash[nf, np.arange[1, len[hashes_at_nf] + 1]] = hashes_at_nf
    assert_(not np.any(np.isnan(F[nf, :])), "Got a NaN. FAIL!")
    # It must be the case hfun values and gradients are the same when called with
    # and without hashes. Because we assume h is cheap to evaluate, we can
    # possibly detect errors in the implementation of hfun by checking each time
    # that the values and gradients agree:
    if tol.hfun_test_mode:
        assert_(iscell(hashes_at_nf), "hfun must return a cell of hashes")
        assert_(np.all(cellfun(ischar, hashes_at_nf)), "Hashes must be character arrays")
        h_dummy1, grad_dummy1, hash_dummy = hfun(F[nf, :])
        h_dummy2, grad_dummy2 = hfun(F[nf, :], hash_dummy)
        assert_(np.any(h_dummy1 == h_dummy2), "hfun values don't agree when being re-called with the same inputs")
        assert_(np.all(np.all(grad_dummy1 == grad_dummy2)), "hfun gradients don't agree when being re-called with the same inputs")

    return nf, X, F, h, Hash, hashes_at_nf

    return nf, X, F, h, Hash, hashes_at_nf
