import numpy as np
from scipy.optimize import linprog
from cvxopt import matrix, solvers
import ipdb

# import matlab # You'll need to uncomment this if you are going to use matlab's linprog below.


def minimize_affine_envelope(f, f_bar, beta, G_k, H, delta, Low, Upp, H_k, subprob_switch, eng):
    solvers.options['show_progress'] = False

    G_k_smaller, cols = np.unique(G_k, axis=1, return_index=True)

    n, p = G_k_smaller.shape

    bk = -(f_bar - f - beta)
    bk_smaller = bk[cols]

    A = np.hstack((-np.ones((p, 1)), G_k_smaller.T))
    ff = np.concatenate((np.array([[1]]), np.zeros((n, 1))))
    x0 = np.vstack((np.array([np.max(-bk_smaller)]), np.zeros((n, 1))))

    assert subprob_switch == "linprog" or subprob_switch == "quadprog", "Unrecognized subprob_switch"

    if subprob_switch == "quadprog":
        # CVXOPT doesn't separate bounds, have to create bounds:
        lower_bound_constraints = np.hstack((np.zeros((n, 1)), -np.eye(n)))
        upper_bound_constraints = np.hstack((np.zeros((n, 1)), np.eye(n)))
        bound_constraints = np.vstack((lower_bound_constraints, upper_bound_constraints))
        A2 = np.vstack((A, bound_constraints))
        A2 = matrix(A2)
        bk_smaller2 = np.concatenate((bk_smaller, -Low))
        bk_smaller2 = np.concatenate((bk_smaller2, Upp))
        bk_smaller2 = matrix(np.expand_dims(bk_smaller2, 1))

        ff = matrix(ff)

        # pad H
        Hmm = np.vstack((np.zeros((1, n)), H))
        Hmm = np.hstack((np.zeros((n + 1, 1)), Hmm))
        Hmm = matrix(Hmm)
        try:
            sol = solvers.qp(Hmm, ff, A2, bk_smaller2)
            assert sol['status'] == 'optimal', "Error in minimize_affine_envelope. Trying rescaling now."
            x = np.array(sol['x'])
            x = x[:(n + 1)]
            duals = np.array(sol['z'])
            duals_g = duals[:p]
            duals_l = duals[p: (p + n)]
            duals_u = duals[(p + n):]
            duals_g = duals_g.squeeze()
            duals_l = duals_l.squeeze()
            duals_u = duals_u.squeeze()

        except Exception as first_exception:
            try:
                normG_k = np.linalg.norm(G_k_smaller)
                A = np.hstack((-np.ones((p, 1)), G_k_smaller.T / normG_k))
                A2 = np.vstack((A, bound_constraints))
                A2 = matrix(A2)
                bk_smaller2 = np.concatenate((bk_smaller, -1.0 * normG_k * Low))
                bk_smaller2 = np.concatenate((bk_smaller2, normG_k * Upp))
                bk_smaller2 = matrix(np.expand_dims(bk_smaller2, 1))
                sol = solvers.qp(Hmm / (normG_k**2), ff, A2, bk_smaller2)
                try:
                    x = np.array(sol['x'])
                    x = x[:(n + 1)]
                    x[1:] = x[1:] / normG_k
                    duals = np.array(sol['z'])
                    duals_g = duals[:p]
                    duals_l = duals[p: (p + n)]
                    duals_u = duals[(p + n):]
                    duals_g = duals_g.squeeze()
                    duals_l = duals_l.squeeze()
                    duals_u = duals_u.squeeze()
                except:
                    return 0, 0, 0, 0, True
            except Exception as second_exception:
                    return 0, 0, 0, 0, True

    if subprob_switch == "linprog":
        options = {"disp": False, "maxiter": (n * p) ** 3, "ipm_optimality_tolerance": 1e-12}
        try:
            res = linprog(c=ff.flatten(), A_ub=A, b_ub=bk_smaller, bounds=list(zip([None] + list(Low), [None] + list(Upp))), options=options, x0=x0, method="highs-ipm")
            assert res["success"], "Error in minimize_affine_envelope. Trying rescaling now."
            x = res.x
            duals_g = -1.0 * res.ineqlin.marginals
            duals_u = -1.0 * res.upper.marginals[1:]
            duals_l = res.lower.marginals[1:]
            #ipdb.set_trace()
            # # Prepare your data for MATLAB. MATLAB expects column-major order,
            # # and the MATLAB Engine API for Python requires MATLAB types for some data
            # c_matlab = matlab.double(ff.flatten().tolist())
            # A_matlab = matlab.double(A.tolist())
            # b_matlab = matlab.double(bk_smaller.tolist())
            # lb_matlab = matlab.double([float('-inf')] + Low.tolist())
            # ub_matlab = matlab.double([float('inf')] + Upp.tolist())
            # # options_matlab = eng.optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'None')
            # options_matlab = eng.optimoptions('linprog', 'Display', 'None')
            # result = eng.linprog(c_matlab, A_matlab, b_matlab, matlab.double([]), matlab.double([]), lb_matlab, ub_matlab, options_matlab, nargout=5)

            # if result[2] == 1: # successful termination
            #     x = np.array([item for sublist in result[0] for item in sublist])
            #     duals_g = np.array([result[4]["ineqlin"]]).squeeze()
            #     duals_u = np.array([item for sublist in result[4]["upper"] for item in sublist])[1:]
            #     duals_l = np.array([item for sublist in result[4]["lower"] for item in sublist])[1:]
            # else:
            #     duals_g = np.zeros(p)
            #     duals_g[0] = 1.0
            #     duals_l = np.zeros(n)
            #     duals_u = np.zeros(n)
            #     x = x0.squeeze()

        except Exception as first_exception:
            print(first_exception)
            try:
                normA = np.linalg.norm(A[:, 1:])
                rescaledA = np.zeros_like(A)
                rescaledA[:, 0] = -np.ones(p)
                rescaledA[:, 1:] = A[:, 1:] / normA
                res = linprog(c=ff.flatten(), A_ub=rescaledA, b_ub=bk_smaller, bounds=list(zip([None] + list(Low), [None] + list(Upp))), options=options, x0=x0, method="highs-ipm")
                assert res["success"], "Error in minimize_affine_envelope. Trying rescaling now."

                x = res.x
                duals_g = -1.0 * res.ineqlin.marginals
                duals_u = -1.0 * res.upper.marginals[1:]
                duals_l = res.lower.marginals[1:]

                # result = eng.linprog(c_matlab, matlab.double(rescaledA.tolist()), b_matlab, matlab.double([]), matlab.double([]), lb_matlab, ub_matlab, x0, options_matlab, nargout=5)

                # if result[2] == 1: # successful termination
                #     x = np.array([item for sublist in result[0] for item in sublist])
                #     duals_g = np.array([result[4]["ineqlin"]]).squeeze()
                #     duals_u = np.array([item for sublist in result[4]["upper"] for item in sublist])[1:]
                #     duals_l = np.array([item for sublist in result[4]["lower"] for item in sublist])[1:]
                # else:
                #     duals_g = np.zeros(p)
                #     duals_g[0] = 1.0
                #     duals_l = np.zeros(n)
                #     duals_u = np.zeros(n)
                #     x = x0.squeeze()

            except Exception as second_exception:
                print(second_exception)
                return 0, 0, 0, 0, True

    lambda_star = np.zeros(G_k.shape[1])
    lambda_star[cols] = duals_g

    s = x[1:]
    if subprob_switch == "quadprog":
        tau = max(-np.expand_dims(bk, 1) + np.dot(G_k.T, s)) + 0.5 * np.dot(s.T, np.dot(H, s))
    elif subprob_switch == "linprog":
        tau = max(-bk + np.dot(G_k.T, s)) + 0.5 * np.dot(s.T, np.dot(H, s))

    if tau > 0:
        tau = 0
        s = np.zeros(n)

    Low[duals_l <= 0] = 0
    Upp[duals_u <= 0] = 0

    chi = np.linalg.norm(np.dot(G_k, lambda_star) - duals_l + duals_u) + np.dot(bk, lambda_star) - np.dot(Low, duals_l) + np.dot(Upp, duals_u)

    return s, tau, chi, lambda_star, False
