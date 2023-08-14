import numpy as np
from scipy.optimize import linprog


def minimize_affine_envelope(f, f_bar, beta, G_k, H, delta, Low, Upp, H_k, subprob_switch):
    G_k_smaller, cols = np.unique(G_k, axis=1, return_index=True)

    n, p = G_k_smaller.shape

    bk = -(f_bar - f - beta)
    bk_smaller = bk[cols]
    H_k_smaller = H_k[cols, :, :]

    A = np.hstack((-np.ones((p, 1)), G_k_smaller.T))
    ff = np.concatenate((np.array([[1]]), np.zeros((n, 1))))
    HH = np.block([[0, np.zeros((1, n))], [np.zeros((n, 1)), H]])
    x0 = np.vstack((np.array([np.max(-bk_smaller)]), np.zeros((n, 1))))

    if subprob_switch == "GAMS_QCP":
        # Implement solve_matts_QCP function here
        pass
    elif subprob_switch == "GAMS_LP":
        # Implement solve_matts_LP function here
        pass
    elif subprob_switch == "linprog":
        options = {"disp": False}
        try:
            res = linprog(c=ff.flatten(), A_ub=A, b_ub=bk_smaller, bounds=list(zip([None] + list(Low), [None] + list(Upp))), options=options, x0=x0)
            x = res.x
            duals_g = res.ineqlin.marginals
            duals_u = res.lower.marginals[1:]
            duals_l = res.upper.marginals[1:]
        except:
            normA = np.linalg.norm(A[:, 1:], axis=0)
            rescaledA = np.zeros_like(A)
            rescaledA[:, 0] = -np.ones(p)
            rescaledA[:, 1:] = A[:, 1:] / normA
            res = linprog(c=ff.flatten(), A_ub=rescaledA, b_ub=bk_smaller, bounds=list(zip([None] + list(Low), [None] + list(Upp))), options=options, x0=x0)
            x = res.x
            duals_g = res.ineqlin.marginals
            duals_u = res.lower.marginals[1:] @ normA
            duals_l = res.upper.marginals[1:] @ normA
    else:
        raise ValueError("Unrecognized subprob_switch")

    lambda_star = np.zeros(G_k.shape[1])
    lambda_star[cols] = duals_g

    s = x[1:]
    tau = max(-bk + np.dot(G_k.T, s)) + 0.5 * np.dot(s.T, np.dot(H, s))
    if tau > 0:
        tau = 0
        s = np.zeros(n)

    Low[duals_l <= 0] = 0
    Upp[duals_u <= 0] = 0

    chi = np.linalg.norm(np.dot(G_k, lambda_star) - duals_l + duals_u) + np.dot(bk.T, lambda_star) - np.dot(Low.T, duals_l) + np.dot(Upp.T, duals_u)

    return s, tau, chi, lambda_star
