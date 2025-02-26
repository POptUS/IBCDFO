from scipy.optimize import minimize
import numpy as np

def objective_for_lbfgsb(y, hfun, hfun_d, Fx, G, H):

    n, m = np.shape(G)
    My = np.zeros(m)
    Jy = np.zeros((n, m))
    grad = np.zeros(n)

    yG = y @ G

    for i in range(m):  # this can certainly be vectorized, I just want it readable for debugging.
        My[i] = Fx[i] + yG[i] + 0.5 * y @ H[:, :, i] @ y.T
        Jy[:, i] = G[:, i] + H[:, :, i] @ y.T

    for j in range(n):
        _, grad[j] = hfun_d(My, Jy[j, :])

    return hfun(My), grad


def run_lbfgsb(hfun, hfun_d, Fx, G, H, L, U):

    #  create wrapper functions (sooooo stupid, but i want to use scipy for now because i trust LBFGS-B)
    def obj(y):
        hFy, _ = objective_for_lbfgsb(y, hfun, hfun_d, Fx, G, H)
        return hFy

    def jac(y):
        _, gradhFy = objective_for_lbfgsb(y, hfun, hfun_d, Fx, G, H)
        return gradhFy

    n, m = np.shape(G)

    x0 = np.zeros(n)

    hFx0 = obj(x0)

    bounds = [(L[i], U[i]) for i in range(n)]
    out = minimize(obj, x0, method='L-BFGS-B', jac=jac, bounds=bounds)
    Xsp = out.x
    success = out.success
    if np.linalg.norm(Xsp) > 0:
        mdec = out.fun - hFx0
    else:  # solver blew it, try backtracking
        g = jac(x0)
        beta = 10
        mdec = 0  # gotta error if the loop below fails
        for j in range(9):
            hftrial = obj(x0 - (beta ** (-j)) * g)
            if hftrial < hFx0:
                out = minimize(obj, x0 - (beta ** (-j)) * g, method='L-BFGS-B', jac=jac, bounds=bounds)
                Xsp = out.x
                success = out.success
                mdec = out.fun - hFx0
                break
    return Xsp, mdec, success
