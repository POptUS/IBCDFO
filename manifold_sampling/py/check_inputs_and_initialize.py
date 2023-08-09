import numpy as np


def check_inputs_and_initialize(x0, F0, nfmax):
    global h_activity_tol
    n = len(x0)
    p = len(F0)
    delta = 0.3
    printf = 1
    h_activity_tol = np.amin(1e-08, delta)
    # Internal parameters/tolerances for formquad
    fq_pars.Par[1] = np.sqrt(n)

    fq_pars.Par[2] = np.amax(10, np.sqrt(n))

    fq_pars.Par[3] = 0.001

    fq_pars.Par[4] = 0.001

    fq_pars.npmax = (n + 1) * (n + 2) / 2

    # Internal parameters/tolerances for manifold sampling
    tol.maxdelta = 1
    tol.mindelta = 1e-13
    tol.gtol = 1e-13
    tol.norm_g_change = 0.001

    tol.kappa_d = 0.0001

    tol.eta1 = 0.01

    tol.eta2 = 10000.0

    tol.eta3 = 0.5
    tol.gamma_dec = 0.5

    tol.gamma_inc = 2

    tol.hfun_test_mode = 1

    # kappa_mh = 0;    # [dbl] > 0 that bounds the component model Hessians

    tol.gentype = 2
    assert_(nfmax >= n + 1, "nfmax is less than n+1, exiting")
    X = np.array([[x0], [np.zeros((nfmax - 1, n))]])

    F = np.array([[F0], [np.zeros((nfmax - 1, p))]])

    h = np.zeros((nfmax, 1))

    Hash = cell(nfmax, 1)

    nf = 1
    trust_rho = 1
    xkin = 1
    Hres = np.zeros((n, n, p))
    return n, delta, printf, fq_pars, tol, X, F, h, Hash, nf, trust_rho, xkin, Hres

    return n, delta, printf, fq_pars, tol, X, F, h, Hash, nf, trust_rho, xkin, Hres
