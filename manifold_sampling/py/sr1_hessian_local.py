import numpy as np
import ipdb
# from .minimize_affine_envelope import minimize_affine_envelope
from .solve_proj_zero_convex_hull import solve_proj_zero_convex_hull

def sr1_hessian_local(H_mm, X, xkin, F, Grad, hfun, Xlist, Hash):

    Xlist = np.array(Xlist).astype(int)
    for i, index in enumerate(Xlist):
        # print(index)
        if index == xkin:
            continue
        s = X[xkin] - X[index]
        Gres_index = Grad[index].T
        Gres_xkin = Grad[xkin].T

        hashes_at_index = Hash[index]
        # hashes_at_xkin = Hash[xkin]

        __, D_k_index = hfun(F[index], hashes_at_index)
        __, D_k_xkin = hfun(F[xkin], hashes_at_index)
    
        G_k_index = Gres_index @ D_k_index
        G_k_xkin = Gres_xkin @ D_k_xkin

        # lmbd = solve_proj_zero_convex_hull(G_k_nf)
        # y = G_k_nf @ lmbd - G_k_xkin @ lmbd
        y = G_k_xkin - G_k_index
        y = np.squeeze(y)

        denominator = (s - y @ H_mm) @ y
        if np.abs(denominator) > 1e-11:
            # H_mm = H_mm + (y - s @ H_mm) @ (y - s @ H_mm).T / denominator
            H_mm = H_mm + (s - y @ H_mm) @ (s - y @ H_mm).T / denominator

        # denominator1 = s @ y
        # # denominator2 = s @ H_mm @ s
        # denominator2 = y @ H_mm @ y
        # # H_mm = H_mm + y.T @ y / denominator1 - (H_mm @ s) @ (H_mm @ s).T / denominator2
        # if np.abs(denominator1) < 1e-11 or np.abs(denominator2) < 1e-11:
        #     H_mm = H_mm
        # else:
        #     H_mm = H_mm + s.T @ s / denominator1 - (H_mm @ y) @ (H_mm @ y).T / denominator2

    return H_mm
