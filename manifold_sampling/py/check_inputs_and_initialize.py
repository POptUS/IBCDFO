import numpy as np


def check_inputs_and_initialize(x0, F0, nf_max, Low, Upp):
    x0 = x0.squeeze()
    n = int(len(x0))
    p = int(len(F0))
    delta = 0.1
    printf = 1

    # Internal parameters/tolerances for formquad

    fq_pars = {"Par": [np.sqrt(n), max(10, np.sqrt(n)), 0.001, 0.001, False], "npmax": 2 * n + 1}

    # Never let the radius exceed half the narrowest side of the box; a larger
    # radius cannot be realised by any feasible step anyway. checkinputss
    # validates the bounds, but only after this runs, so fall back to the
    # delta-relative ceiling when they do not conform and leave the
    # complaining to it.
    box_is_valid = np.shape(Low) == np.shape(Upp) and np.min(Upp - Low) > 0
    if box_is_valid:
        maxdelta = min(1.0e3 * delta, 0.5 * np.min(Upp - Low))
    else:
        maxdelta = 1.0e3 * delta
    # An infeasible or degenerate box is checked by checkinputss.
    # Flag a maxdelta that is too small when the box itself is valid.
    if box_is_valid:
        assert maxdelta >= delta, f"maxdelta ({maxdelta:g}) is less than delta ({delta:g}); the box may be too narrow for delta"

    # Internal parameters/tolerances for manifold sampling
    tol = {
        "maxdelta": maxdelta,
        "mindelta": 1e-13,
        "gtol": 1e-13,
        "norm_g_change": 0.001,
        "kappa_d": 0.0001,
        "eta1": 0.01,
        "eta2": 10000.0,
        "eta3": 0.5,
        "gamma_dec": 0.5,
        "gamma_inc": 2,
        "hfun_test_mode": 1,
        "snap_tol": 1.0e-6,
    }

    # kappa_mh = 0;    # [dbl] > 0 that bounds the component model Hessians

    assert nf_max >= n + 1, "nf_max is less than n+1, exiting"

    X = np.vstack((x0, np.zeros((nf_max - 1, n))))
    F = np.vstack((F0, np.zeros((nf_max - 1, p))))
    hF = np.zeros((nf_max, 1))
    Hash = {}

    nf = 0
    successful = 1
    xk_in = 0
    Hres = np.zeros((n, n, p))
    chi_k = np.inf

    return n, delta, printf, fq_pars, tol, X, F, hF, Hash, nf, successful, xk_in, Hres, chi_k
