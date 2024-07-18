import numpy as np
import ipdb

def overapproximating_hessian(H_mm, X, nf, xkin, h, G_k, f_bar, beta, delta):

    # hyperparameter that should technically be tunable, must be greater than 0.5:
    rho = 0.5 + 1e-8

    f = h[xkin]

    # Initialize overapproximating_hessian with something near 0:
    n, p = G_k.shape

    # Find all points within delta of X[xkin]
    for trial in range(nf):
        d = X[xkin] - X[trial]
        if np.linalg.norm(d) <= delta and xkin != trial:
            # calculate current model value:
            maxlinear = d @ G_k + f_bar - f * np.ones(p) - beta

            hessian_contribution = d @ H_mm @ d
            md = f + np.amax(maxlinear) + 0.5 * hessian_contribution

            if md < h[trial]:
                # need to increase hessian in direction d
                correction = np.minimum(h[trial] - md + 0.5 * hessian_contribution, rho * hessian_contribution)
                multiplier = -1.0 + np.sqrt(2.0 * correction / hessian_contribution)

                normalized_d = d / np.linalg.norm(d)
                rank_one_update = np.eye(n) + multiplier * normalized_d.T @ normalized_d

                H_mm = rank_one_update @ H_mm @ rank_one_update

    return H_mm
