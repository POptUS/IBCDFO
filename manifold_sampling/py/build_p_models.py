from ibcdfo.pounders import formquad

from .evaluate_points_to_force_valid_model import evaluate_points_to_force_valid_model
from .update_models import update_models


def build_p_models(nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U):
    n = X.shape[1]
    p = F.shape[1]

    valid, Gres, Hres, X, F, h, nf, Hash = update_models(hfun, Ffun, n, p, nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, 1, Hash, tol, L, U)
    # Evaluate model-improving points if necessary
    if not valid and not len(Gres) == 0:
        Mdir, np, valid, *_ = formquad(X[: nf + 1], F[: nf + 1], delta, xkin, fq_pars["npmax"], fq_pars["Par"], 1)
        if valid:
            raise Exception("what to do here")
        X, F, h, nf, Hash = evaluate_points_to_force_valid_model(n, nf, xkin, delta, X, F, h, tol["gentype"], Mdir, np, hfun, Ffun, Hash, fq_pars, tol, nfmax, L, U)
        __, Gres, Hres, X, F, h, nf, Hash = update_models(hfun, Ffun, n, p, nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, 1, Hash, tol, L, U)

    return Gres, Hres, X, F, h, nf, Hash


def build_p_models2(nf, nfmax, xkin, delta, F, G, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U):
    """
    --INPUTS-----------------------------------------------------------------
    nf      [int] # of function evals so far (>n+1) (.5*(n+1)*(n+2))
    nfmax   [int] Max # function evals (>n+1) (.5*(n+1)*(n+2))
    xkin    [int] Index in (X and F) of the current center
    delta   [dbl] Positive trust region radius
    F       [dbl] [nf-by-m] Function values of evaluated points
    G       [dbl] [nf-by-n-by-m] Function values of evaluated points
    X       [dbl] [nf-by-n] Locations of evaluated points
    h       [dbl] [nf-by-nh] h(F(x)) values
    Hres    [dbl] [nf-by-nh] Current model Hessians
    fq_pars [dic] Dictionary of form_quad parameters
    tol     [dbl] [4] More form_quad parameters
    hfun    [fun] Function for evaluating H
    Ffun    [fun] Function for evaluating F
    Hash    [int] [nf-by-m] array of manifold activities at points in X
    L       [dbl] [n] Lower bound array
    U       [dbl] [n] Upper bound array
    --OUTPUTS----------------------------------------------------------------
    valid   [log] Flag saying if model is valid within Pars[2]*delta
    Gres    [dbl] [n-by-m]  Matrix of model gradients at Xk
    Hres    [dbl] [n-by-n-by-m]  Array of model Hessians at Xk
    X       [dbl] [nf-by-n] Locations of evaluated points
    F       [dbl] [nf-by-m] Function values of evaluated points
    h       [dbl] [nf-by-nh] h(F(x)) values
    nf      [int] # of function evals so far (>n+1) (.5*(n+1)*(n+2)) after finishing
    Hash    [int] [nf-by-m] Updated array of manifold activities at points in X
    """

    n = X.shape[1]
    p = F.shape[1]

    valid, Gres, Hres, X, F, h, nf, Hash = update_models(hfun, Ffun, n, p, nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, 1, Hash, tol, L, U)
    # Evaluate model-improving points if necessary
    if not valid and not len(Gres) == 0:
        Mdir, np, valid, *_ = formquad(X[: nf + 1], F[: nf + 1], delta, xkin, fq_pars["npmax"], fq_pars["Par"], 1)
        if valid:
            raise Exception("what to do here")
        X, F, h, nf, Hash = evaluate_points_to_force_valid_model(n, nf, xkin, delta, X, F, h, tol["gentype"], Mdir, np, hfun, Ffun, Hash, fq_pars, tol, nfmax, L, U)
        __, Gres, Hres, X, F, h, nf, Hash = update_models(hfun, Ffun, n, p, nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, 1, Hash, tol, L, U)

    return Gres, Hres, X, F, G, h, nf, Hash
