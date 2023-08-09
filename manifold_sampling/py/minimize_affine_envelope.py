import numpy as np


def minimize_affine_envelope(f, f_bar, beta, G_k, H, delta, Low, Upp, H_k, subprob_switch):
    G_k_smaller, cols = uniquetol(np.transpose(G_k), "ByRows", True)
    G_k_smaller = np.transpose(G_k_smaller)
    n, p = G_k_smaller.shape
    bk = -(np.transpose(f_bar) - f - beta)
    bk_smaller = bk[cols]
    H_k_smaller = H_k[cols, :, :]
    # check first if mu = 0 (i.e. TR is not active)
    A = np.array([-np.ones((p, 1)), np.transpose(G_k_smaller)])
    ff = np.array([[1], [np.zeros((n, 1))]])
    HH = np.array([[0, np.zeros((1, n))], [np.zeros((n, 1)), H]])
    x0 = np.array([[np.amax(-bk_smaller)], [np.zeros((n, 1))]])
    if str(subprob_switch) == str("GAMS_QCP"):
        x, duals_g, duals_u, duals_l = solve_matts_QCP(ff, A, bk_smaller, x0, delta, Low, Upp, H_k_smaller)
        duals_u = duals_u(np.arange(2, n + 1 + 1))
        duals_l = duals_l(np.arange(2, n + 1 + 1))
    else:
        if str(subprob_switch) == str("GAMS_LP"):
            x, duals_g, duals_u, duals_l = solve_matts_LP(ff, A, bk_smaller, x0, Low, Upp)
            duals_u = duals_u(np.arange(2, n + 1 + 1))
            duals_l = duals_l(np.arange(2, n + 1 + 1))
        else:
            if str(subprob_switch) == str("linprog"):
                options = optimoptions("linprog", "Display", "none")
                try:
                    x, __, exitflag, __, lambda_ = linprog(ff, A, bk_smaller, [], [], np.transpose(np.array([-Inf, Low])), np.transpose(np.array([Inf, Upp])), options)
                    if exitflag == 1:
                        duals_g = lambda_.ineqlin
                        duals_u = lambda_.upper[np.arange(2, n + 1 + 1)]
                        duals_l = lambda_.lower[np.arange(2, n + 1 + 1)]
                    else:
                        duals_g = np.zeros((p, 1))
                        duals_g[1] = 1.0
                        duals_l = np.zeros((n, 1))
                        duals_u = np.zeros((n, 1))
                        x = x0
                finally:
                    pass
            else:
                raise Exception("Unrecognized subprob_switch")

    lambda_star = sparse(G_k.shape[2 - 1], 1)
    lambda_star[cols] = duals_g
    s = x[np.arange(2, end() + 1)]
    tau = np.amax(-bk + np.transpose(G_k) * s) + 0.5 * np.transpose(s) * H * s
    if tau > 0:
        # something went wrong
        tau = 0
        s = np.zeros((n, 1))

    Low[duals_l <= 0] = 0
    Upp[duals_u <= 0] = 0
    chi = np.linalg.norm(G_k * lambda_star - duals_l + duals_u) + np.transpose(bk) * lambda_star - Low * duals_l + Upp * duals_u
    return s, tau, chi, lambda_star

    return s, tau, chi, lambda_star
