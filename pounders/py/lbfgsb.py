from scipy.optimize import minimize
import numpy as np
import ipdb

def objective_for_lbfgsb(y, hfun, hfun_d, Fx, G, H):

    n, m = np.shape(G)
    My = np.zeros(m)
    Jy = np.zeros((n, m))

    yG = y @ G

    for i in range(m):  # this can certainly be vectorized, I just want it readable for debugging.
        My[i] = Fx[i] + yG[i] + 0.5 * y @ H[:, :, i] @ y.T
        Jy[:, i] = G[:, i] + H[:, :, i] @ y.T

    return hfun(My), hfun_d(My, Jy)


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
    #x0 = np.random.uniform(L, U)

    hFx0 = obj(x0)

    bounds = [(L[i], U[i]) for i in range(n)]
    options = {"gtol": 1e-8, "ftol": 1e-8}
    out = minimize(obj, x0, method='L-BFGS-B', jac=jac, bounds=bounds, options=options)
    Xsp = out.x
    success = out.success
    fval = obj(Xsp)
    if np.linalg.norm(Xsp) > 0 and fval < hFx0:
        mdec = fval - hFx0
        return Xsp, mdec, success
    else:
        # try running LBFGS without the explicit gradient.
        print("Need to do an LBFGS run without gradients because of gradient failure!")
        out = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, options=options)
        success = out.success
        Xsp = out.x
        fval = obj(Xsp)
        if np.linalg.norm(Xsp) > 0 and fval < hFx0:
            mdec = out.fun - hFx0
        else:  # solver blew it, try backtracking
            print("Trying a backtrack now because LBFGS without gradients also blew it!")
            g = jac(x0)
            beta = 10
            mdec = 0  # gotta error if the loop below fails
            for j in range(9):
                trial = x0 - (beta ** (-j)) * g
                hftrial = obj(trial)
                if hftrial < hFx0:
                    success = out.success
                    Xsp = trial - x0
                    mdec = hftrial - hFx0
                    break
        return Xsp, mdec, success
