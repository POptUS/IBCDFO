import numpy as np
from scipy.optimize import linprog


def minimize_affine_envelope(f, f_bar, beta, G_k, H, delta, Low, Upp, H_k, subprob_switch):
    G_k_smaller, cols = np.unique(G_k, axis=1, return_index=True)

    n, p = G_k_smaller.shape

    bk = -(f_bar - f - beta)
    bk_smaller = bk[cols]

    A = np.hstack((-np.ones((p, 1)), G_k_smaller.T))
    ff = np.concatenate((np.array([[1]]), np.zeros((n, 1))))
    x0 = np.vstack((np.array([np.max(-bk_smaller)]), np.zeros((n, 1))))

    assert subprob_switch == "linprog", "Unrecognized subprob_switch"

    if subprob_switch == "linprog":
        options = {"disp": False}
        try:
            res = linprog(c=ff.flatten(), A_ub=A, b_ub=bk_smaller, bounds=list(zip([None] + list(Low), [None] + list(Upp))), options=options, x0=x0)
            x = res.x
            duals_g = -1.0 * res.ineqlin.marginals
            duals_u = -1.0 * res.upper.marginals[1:]
            duals_l = res.lower.marginals[1:]
        except Exception as e:
            print(e)
            normA = np.linalg.norm(A[:, 1:])
            rescaledA = np.zeros_like(A)
            rescaledA[:, 0] = -np.ones(p)
            rescaledA[:, 1:] = A[:, 1:] / normA
            res = linprog(c=ff.flatten(), A_ub=rescaledA, b_ub=bk_smaller, bounds=list(zip([None] + list(Low), [None] + list(Upp))), options=options, x0=x0)
            x = res.x
            duals_g = -1.0 * res.ineqlin.marginals
            duals_u = -1.0 * res.upper.marginals[1:] * normA
            duals_l = res.lower.marginals[1:] * normA

    lambda_star = np.zeros(G_k.shape[1])
    lambda_star[cols] = duals_g

    s = x[1:]
    tau = max(-bk + np.dot(G_k.T, s)) + 0.5 * np.dot(s, np.dot(H, s))
    if tau > 0:
        tau = 0
        s = np.zeros(n)

    Low[duals_l <= 0] = 0
    Upp[duals_u <= 0] = 0

    chi = np.linalg.norm(np.dot(G_k, lambda_star) - duals_l + duals_u) + np.dot(bk, lambda_star) - np.dot(Low, duals_l) + np.dot(Upp, duals_u)

    return s, tau, chi, lambda_star
