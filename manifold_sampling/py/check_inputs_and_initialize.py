import numpy as np


def check_inputs_and_initialize(x0, p, nf_max, Options):
    x0 = x0.squeeze()
    n = int(len(x0))
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

    successful = 1
    Hres = np.zeros((n, n, p))
    chi_k = np.inf

    return n, delta, printf, fq_pars, tol, successful, Hres, chi_k
