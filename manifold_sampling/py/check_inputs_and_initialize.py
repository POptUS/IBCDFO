import numpy as np


def check_inputs_and_initialize(x0, F0, nf_max):
    x0 = x0.squeeze()
    n = int(len(x0))
    p = int(len(F0))
    delta = 0.1
    printf = 1

    # Internal parameters/tolerances for formquad

    fq_pars = {"Par": [np.sqrt(n), max(10, np.sqrt(n)), 0.001, 0.001, False], "npmax": (n + 1) * (n + 2) // 2}

    # Internal parameters/tolerances for manifold sampling
    tol = {
        "maxdelta": 1,
        "mindelta": 1e-8,
        "gtol": 1e-8,
        "norm_g_change": 0.001,
        "kappa_d": 0.0001,
        "eta1": 0.01,
        "eta2": 10000.0,
        "eta3": 0.5,
        "gamma_dec": 0.5,
        "gamma_inc": 2,
        "hfun_test_mode": 1,
        "gentype": 3,
    }

    # kappa_mh = 0;    # [dbl] > 0 that bounds the component model Hessians

    assert nf_max >= n + 1, "nf_max is less than n+1, exiting"

    X = np.vstack((x0, np.zeros((nf_max - 1, n))))
    F = np.vstack((F0, np.zeros((nf_max - 1, p))))
    h = np.zeros((nf_max, 1))
    Hash = {}

    nf = 0
    trust_rho = 1
    xkin = 0
    Hres = np.zeros((n, n, p))

    return n, delta, printf, fq_pars, tol, X, F, h, Hash, nf, trust_rho, xkin, Hres
