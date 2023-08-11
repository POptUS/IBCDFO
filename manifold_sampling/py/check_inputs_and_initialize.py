import numpy as np


def check_inputs_and_initialize(x0, F0, nfmax):
    global h_activity_tol
    n = len(x0)
    p = len(F0)
    delta = 0.3
    printf = 1
    h_activity_tol = min(1e-08, delta)

    # Internal parameters/tolerances for formquad

    fq_pars = {'Par': {1: np.sqrt(n), 2: max(10, np.sqrt(n)), 3: 0.001, 4: 0.001}, 'npmax': (n + 1) * (n + 2) / 2}

    # Internal parameters/tolerances for manifold sampling
    tol = {
    'maxdelta' : 1,
    'mindelta' : 1e-13,
    'gtol' : 1e-13,
    'norm_g_change' : 0.001,
    'kappa_d' : 0.0001,
    'eta1' : 0.01,
    'eta2' : 10000.0,
    'eta3' : 0.5,
    'gamma_dec' : 0.5,
    'gamma_inc' : 2,
    'hfun_test_mode' : 1,
    }

    # kappa_mh = 0;    # [dbl] > 0 that bounds the component model Hessians

    tol['gentype'] = 2

    assert nfmax >= n + 1, "nfmax is less than n+1, exiting"

    X = np.vstack((x0, np.zeros((nfmax - 1, n))))
    F = np.vstack((F0, np.zeros((nfmax - 1, p))))
    h = np.zeros((nfmax, 1))
    Hash = {}

    nf = 0
    trust_rho = 1
    xkin = 0
    Hres = np.zeros((n, n, p))

    return n, delta, printf, fq_pars, tol, X, F, h, Hash, nf, trust_rho, xkin, Hres
