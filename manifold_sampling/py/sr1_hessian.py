import numpy as np
import ipdb
# from .minimize_affine_envelope import minimize_affine_envelope
from .solve_proj_zero_convex_hull import solve_proj_zero_convex_hull

def sr1_hessian(H_mm, X, nf, xkin, F, Grad, hfun, Hash):

    s = X[nf] - X[xkin]

    Gres_nf = Grad[nf].T
    Gres_xkin = Grad[xkin].T

    hashes_at_nf = Hash[nf]
    hashes_at_xkin = Hash[xkin]

    __, D_k_nf = hfun(F[nf], hashes_at_nf)
    __, D_k_xkin = hfun(F[xkin], hashes_at_nf)
    
    G_k_nf = Gres_nf @ D_k_nf
    G_k_xkin = Gres_xkin @ D_k_xkin

    # lmbd = solve_proj_zero_convex_hull(G_k_nf)
    # y = G_k_nf @ lmbd - G_k_xkin @ lmbd
    y = G_k_nf - G_k_xkin
    y = np.squeeze(y)

    denominator = (y - s @ H_mm) @ s
    # denominator = (s - y @ H_mm) @ y
    if np.abs(denominator) > 1e-11:
        H_mm = H_mm + (y - s @ H_mm) @ (y - s @ H_mm).T / denominator
        # H_mm = H_mm + (s - y @ H_mm) @ (s - y @ H_mm).T / denominator


    return H_mm
