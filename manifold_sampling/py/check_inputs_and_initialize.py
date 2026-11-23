import numpy as np


def check_inputs_and_initialize(x0, F0, nf_max, Options):
    x0 = x0.squeeze()
    n = int(len(x0))
    p = int(len(F0))
    delta = Options["delta_0"]
    printf = Options["printf"]

    # Internal parameters/tolerances for formquad

    fq_pars = {"Par": [np.sqrt(n), max(10, np.sqrt(n)), 0.001, 0.001, False], "npmax": 2 * n + 1}

    # Internal parameters/tolerances for manifold sampling
    tol = {
        "maxdelta": Options["delta_max"],
        "mindelta": Options["delta_min"],
        "gtol": Options["g_tol"],
        "norm_g_change": 0.001,
        "kappa_d": 0.0001,
        "eta1": Options["eta_1"],
        "eta2": 10000.0,
        "eta3": Options["eta_3"],
        "gamma_dec": Options["gamma_dec"],
        "gamma_inc": Options["gamma_inc"],
        "hfun_test_mode": Options["hfun_test_mode"],
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
