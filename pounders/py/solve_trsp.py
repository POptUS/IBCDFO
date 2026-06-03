import sys
from functools import lru_cache

import numpy as np

#from .._get_minq_installation import get_minq_installation
from .bqmin import bqmin

from scipy.optimize import minimize


@lru_cache(maxsize=1)
def _get_minqsw():
    #required_minq_SHA, minq_installation = get_minq_installation()

    #if not minq_installation["is_valid"]:
    #    msg = f"Please set MINQ clone to git commit {required_minq_SHA}.\nSee User Guide (https://ibcdfo.readthedocs.io) for more information and instructions."
    #    sys.exit(msg)

    from minqsw import minqsw

    return minqsw

def objective_for_lbfgsb(y, hfun, hfun_d, Fx, G, H, compute_grad=False, regularizer=0.0):

    n, m = np.shape(G)
    My = np.zeros(m)
    if compute_grad:
        Jy = np.zeros((n, m))

    yG = y @ G

    for i in range(m):  # this can certainly be vectorized, I just want it readable for debugging.
        My[i] = Fx[i] + yG[i] + 0.5 * y @ H[:, :, i] @ y.T
        if compute_grad:
            Jy[:, i] = G[:, i] + H[:, :, i] @ y.T

    if compute_grad:
        hfundMy = hfun_d(My)
        grad = Jy @ hfundMy + regularizer * y.T
        return grad
    else:
        hfunMy = hfun(My) + 0.5 * regularizer * (y @ y.T)
        return hfunMy


def run_lbfgsb(hfun, hfun_d, Fx, G, H, L, U, initial_point=None, regularize=False, regularizer=None):

    if not regularize:
        regularizer = 0.0

    #  create wrapper functions (sooooo stupid, but i want to use scipy for now because i trust LBFGS-B)
    def obj(y):
        hFy = objective_for_lbfgsb(y, hfun, hfun_d, Fx, G, H, compute_grad=False, regularizer=regularizer)
        return hFy

    def jac(y):
        gradhFy = objective_for_lbfgsb(y, hfun, hfun_d, Fx, G, H, compute_grad=True, regularizer=regularizer)
        return gradhFy

    n, m = np.shape(G)

    if initial_point is None:
        x0 = np.zeros(n)
    else:
        x0 = initial_point

    hFx0 = obj(x0)

    bounds = [(L[i], U[i]) for i in range(n)]
    options = {"gtol": 1e-12, "ftol": 1e-12}
    #print("Remember: You turned off gradients for now until you fix them.")
    out = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, options=options, jac=jac)
    Xsp = out.x
    success = out.success
    fval = obj(Xsp)
    mdec = fval - hFx0
    return Xsp, mdec, success


def solve_trsp(H, G, Cres, Hres, Gres, hfun, hfun_d, Low, Upp, xk, delta, spsolver, n):
    """
    Solve the bound-constrained trust-region subproblem.

    min  G.T * s + 0.5 * s.T * H * s
    s.t. max(Low - xk, -delta) <= s <= min(Upp - xk, delta)
    """

    Lows = np.maximum(Low - xk, -delta * np.ones(np.shape(Low)))
    Upps = np.minimum(Upp - xk, delta * np.ones(np.shape(Upp)))

    if spsolver == 1:
        Xsp, mdec = bqmin(H, G, Lows, Upps)
        return Xsp, mdec, 0

    if spsolver == 2:
        minqsw = _get_minqsw()
        Xsp, mdec, minq_err, _ = minqsw(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
        if minq_err < 0:
            return Xsp, mdec, -4
        return Xsp, mdec, 0

    if spsolver == 3:
        Xsp, mdec, success = run_lbfgsb(hfun, hfun_d, Cres, Gres, Hres, Lows.T, Upps.T, initial_point=None)
        # need to go check docs for error codes on LBFGSB, return error flag if something went very wrong
        return Xsp, mdec, success

    if spsolver == 4:
        Xsp, mdec, success = run_lbfgsb(hfun, hfun_d, Cres, Gres, np.zeros_like(Hres), Lows.T, Upps.T,
                                        initial_point=None)
        # need to go check docs for error codes on LBFGSB, return error flag if something went very wrong
        return Xsp, mdec, success

    if spsolver == 5:
        # This is what the theory says we should be doing.
        # hardcoded for now (values taken from Conn, Scheinberg, Zhang)
        kappa1 = 1.0
        kappa2 = 1.0
        kappa3 = 0.01

        c = hfun(Cres) ** 2
        regularize = False

        normg = np.linalg.norm(G)
        if normg >= kappa1:
            Hres = np.zeros_like(Hres)
        elif normg < kappa1 and c < kappa2 * normg:
            Hres = np.zeros_like(Hres)
            regularize = True

        Xsp, mdec, success = run_lbfgsb(hfun, hfun_d, Cres, Gres, Hres, Lows.T, Upps.T,
                                        initial_point=None, regularize=regularize, regularizer=(kappa3 * np.sqrt(hfun(Cres))))
        # need to go check docs for error codes on LBFGSB, return error flag if something went very wrong
        return Xsp, mdec, success


    raise ValueError(f"Unknown trust-region subproblem solver: {spsolver}")
