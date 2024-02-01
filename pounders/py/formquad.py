import numpy as np
import scipy.linalg

from .phi2eval import phi2eval
from .bmpts import bmpts


# from .flipFirstRow import flipFirstRow
# from .flipSignQ import flipSignQ


class NanValueError(Exception):
    pass


class ModelBuildingError(Exception):
    pass


def formquad(X, F, delta, xk_in, np_max, Pars, vf):
    """
    formquad(X, F, delta, xk_in, np_max, Pars, vf) -> [Mdir, np, valid, G, H, Mind]
    Computes the parameters for m quadratics
        ### FIX COMMENT Line 15 ###
        Q_i(x) = C(i) + G(:,i)'*x + 0.5*x'*H(:,:,i)*x,  i=1:m
    whose Hessians H are of least Frobenius norm subject to the interpolation
        Q_i(X[Mind,:]) = F[Mind,i].
    The procedure works equally well with m=1 and m>1.
    The derivation is involved but may be found in "MNH: A Derivative-Free
    Optimization Algorithm Using Minimal Norm Hessians" by S Wild, 2008.
    --INPUTS-----------------------------------------------------------------
    X       [dbl] [nf-by-n] Locations of evaluated points
    F       [dbl] [nf-by-m] Function values of evaluated points
    delta   [dbl] Positive trust region radius
    xk_in    [int] Index in (X and F) of the current center
    np_max   [int] Max # interpolation points (>=n+1) (.5*(n+1)*(n+2))
    Pars[0] [dbl] delta multiplier for checking validity
    Pars[1] [dbl] delta multiplier for all interpolation points
    Pars[2] [dbl] Pivot threshold for validity
    Pars[3] [dbl] Pivot threshold for additional points (.001)
    vf      [log] Flag indicating you just want to check model validity
    --OUTPUTS----------------------------------------------------------------
    Mdir    [dbl] [(n-np+1)-by-n]  Unit directions to improve model
    np      [int] Number of interpolation points (=length(Mind))
    valid   [log] Flag saying if model is valid within Pars[2]*delta
    G       [dbl] [n-by-m]  Matrix of model gradients centered at X[xkin]
    H       [dbl] [n-by-n-by-m]  Array of model Hessians centered at X[xkin]
    Mind    [int] [npmax-by-1] Integer vector of model interpolation indices
    """
    # % --DEPENDS ON-------------------------------------------------------------
    # phi2eval : Evaluates the quadratic basis for vector inputs
    # qrinsert, svd : scipy.linalg and numpy
    # Internal parameters:
    nf, n = np.shape(X)
    m = np.shape(F)[1]
    G = np.zeros((n, m))
    H = np.zeros((n, n, m))
    # Precompute the scaled displacements (could be expensive for larger nf_max)
    D = np.zeros((nf, n))  # Scaled displacements
    scale_mat = np.ones((n, n)) / np.sqrt(2)
    scale_mat[np.diag_indices(n)] = 1
    inds_to_use_in_H = np.triu_indices(n)

    assert isinstance(np_max, int), "Must be an integer"
    assert isinstance(xk_in, int), "Must be an integer"

    D = (X[:nf] - X[xk_in]) / delta
    Nd = np.linalg.norm(D, 2, axis=1)

    # Get n+1 sufficiently affinely independent points:
    # Initialize the QR factorization of interest
    Q = np.eye(n)
    R = np.empty(shape=(0, 0))
    # Indices of model interpolation points
    Mind = [xk_in]
    valid = False
    # Counter for number of interpolation points
    mp = 0
    for aff in range(2):
        for i in reversed(range(nf)):
            if Nd[i] <= Pars[aff]:
                proj = np.linalg.norm(D[i] @ Q[:, mp:n], 2)  # Project D onto null
                if proj >= Pars[aff + 2]:  # add this index to Mind
                    mp += 1
                    Mind.append(i)
                    if np.shape(R)[0] == 0:
                        [Q, R] = np.linalg.qr(D[[i], :].T, mode="complete")
                        # [Q, R] = flipFirstRow(Q, R, 0, np.shape(Q)[1]-1)
                        # [Q, R] = flipSignQ(Q, R, 0, np.shape(Q)[1]-1)
                    else:
                        # Update QR
                        D[i] = np.float64(D[i])  # Convert entries to float to use qr_insert
                        [Q, R] = scipy.linalg.qr_insert(Q, R, D[[i], :].T, mp - 1, "col")
                        # [Q, R] = flipFirstRow(Q, R, 0, np.shape(Q)[1]-1)
                        # [Q, R] = flipSignQ(Q, R, 0, np.shape(Q)[1]-1)
                    if mp == n:
                        break  # Breaks out for loop
        if aff == 0 and mp == n:  # Have enough points
            # Mdir = [] - might need to uncomment if next line breaks
            Mdir = np.empty(shape=(0, 0))
            valid = True
            break
        elif aff == 1 and mp < n:  # Need to evaluate more points, then recall
            Mdir = Q[:, mp:n].T  # Output Geometry directions
            G = np.empty(shape=(0, 0))
            H = np.empty(shape=(0, 0))
            return [Mdir, mp, valid, G, H, Mind]
        elif aff == 0:  # Output model-improving directions
            Mdir = Q[:, mp:n].T  # Will be empty if mp=n
        if vf:  # Only needed to do validity check
            return [Mdir, mp, valid, G, H, Mind]
    # Collect additional points
    N = phi2eval(D[Mind]).T

    mp = len(Mind)
    M = np.vstack((np.ones(n + 1), D[Mind].T))
    [Q, R] = np.linalg.qr(M.T, mode="complete")
    # [Q, R] = flipSignQ(Q, R, 0, np.shape(Q)[1]-1)
    # Now we add points until we have np_max starting with the most recent ones
    i = nf - 1
    while mp < np_max or np_max == n + 1:
        if Nd[i] <= Pars[1] and i not in Mind:
            Ny = np.hstack((N, phi2eval(D[[i], :]).T))
            # Update QR
            D[i] = np.float64(D[i])  # Convert entries to float to use qr_insert
            [Qy, Ry] = scipy.linalg.qr_insert(Q, R, np.hstack((1, D[i])), mp, "row")
            # [Qy, Ry] = flipFirstRow(Qy, Ry, 0, np.shape(Q)[1]-1)
            # [Qy, Ry] = flipSignQ(Qy, Ry, 0, np.shape(Q)[1]-1)
            Ly = Ny @ Qy[:, n + 1 : mp + 1]
            s = np.linalg.svd(Ly, compute_uv=False)
            if min(s) > Pars[3]:
                mp += 1
                Mind.append(i)
                N = Ny
                Q = Qy
                R = Ry
                L = Ly
                Z = Q[:, n + 1 : mp]
                # Note that M is growing
                M = np.hstack((M, np.vstack((1, D[[i]].T))))
        i -= 1
        # Reached end of points
        if i == -1:
            # Set outputs so that Hessian is zero
            if mp == n + 1:
                # L = 1
                L = np.array([[1]])
                Z = np.zeros((n + 1, int(0.5 * n * (n + 1))))
                N = np.zeros((int(0.5 * n * (n + 1)), n + 1))
            break
    F = F[Mind]
    for k in range(m):
        # For L = N * Z, solve L.T * L * Omega = Z.T * f:
        J = Z.T @ F[:, [k]]
        if np.shape(L)[1] != np.shape(J)[1]:
            Omega = np.linalg.solve(L.T @ L, J)
        else:
            Omega = np.linalg.solve(L.T @ L, J.T)
        Beta = L @ Omega
        if np.shape(Beta)[1] > 1:
            Beta = Beta.T
        if np.shape(M)[0] == np.shape(M)[1]:
            Alpha = np.linalg.solve(M.T, F[:, [k]] - N.T @ Beta)
        else:
            if np.shape(M)[0] != np.shape(N)[1]:
                Alpha = np.linalg.lstsq(M.T, F[:, [k]] - N.T @ Beta, rcond=None)[0]
            else:
                Alpha = np.linalg.lstsq(M, F[:, [k]] - N.T @ Beta, rcond=None)[0]
            Alpha = np.reshape(Alpha, (np.shape(Alpha)[0], 1))
        G[:, k] = Alpha[1 : n + 1, 0]

        H[:, :, k][inds_to_use_in_H] = Beta.squeeze()  # Set diagonal and above to Beta
        H[:, :, k].T[inds_to_use_in_H] = Beta.squeeze()  # Set diagonal and below to Beta
        H[:, :, k] *= scale_mat  # Scale Beta

    H = H / (delta**2)
    G = G / delta

    return [Mdir, mp, valid, G, H, Mind]


def build_formquad_models(Cres, Hres, Model, delta, n, Low, Upp, X, F, hF, nf, xk_in, nf_max, Ffun, hfun, printf):
    Res = np.zeros(np.shape(F))

    #  1a. Compute the interpolation set.
    D = X[: nf + 1] - X[xk_in]
    Res[: nf + 1, :] = (F[: nf + 1, :] - Cres) - np.diagonal(0.5 * D @ (np.tensordot(D, Hres, axes=1))).T
    [Mdir, mp, valid, Gres, Hresdel, Mind] = formquad(X[0 : nf + 1, :], Res[0 : nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], 0)
    if mp < n:
        [Mdir, mp] = bmpts(X[xk_in], Mdir[0 : n - mp, :], Low, Upp, delta, Model["Par"][2])
        for i in range(int(min(n - mp, nf_max - (nf + 1)))):
            nf += 1
            X[nf] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Mdir[i, :]))
            F[nf] = Ffun(X[nf])
            if np.any(np.isnan(F[nf])):
                raise NanValueError("NaN encountered")
            hF[nf] = hfun(F[nf])
            if printf:
                print("%4i   Geometry point  %11.5e\n" % (nf, hF[nf]))
            D = Mdir[i, :]
            Res[nf, :] = (F[nf, :] - Cres) - 0.5 * D @ np.tensordot(D.T, Hres, 1)
        if nf + 1 >= nf_max:
            raise MaxEvalError("All done")
        [_, mp, valid, Gres, Hresdel, Mind] = formquad(X[0 : nf + 1, :], Res[0 : nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], False)
        if mp < n:
            raise ModelBuildingError("yeah")

    #  1b. Update the quadratic model
    Cres = F[xk_in]
    Hres = Hres + Hresdel

    return Cres, Gres, Hres, nf, X, F, hF, Mind, valid, mp


def formquad_model_improvement(nf, nf_max, valid, rho, eta_1, X, F, hF, delta, xk_in, Model, Cres, Gres, Hres, combinemodels, n, Low, Upp, Ffun, hfun, printf):
    Res = np.zeros(np.shape(F))

    # Need to check because model may be valid after Xsp evaluation
    [Mdir, mp, valid, _, _, _] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], 1)
    if not valid:  # ! One strategy for choosing model-improving point:
        # Update model (exists because delta & xk_in unchanged)
        D = X[: nf + 1] - X[xk_in]
        Res[: nf + 1, :] = (F[: nf + 1, :] - Cres) - np.diagonal(0.5 * D @ (np.tensordot(D, Hres, axes=1))).T
        [_, _, valid, Gres, Hresdel, Mind] = formquad(X[: nf + 1, :], Res[: nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], False)
        Hres = Hres + Hresdel
        # Update for modelimp; Cres unchanged b/c xk_in unchanged
        G, H = combinemodels(Cres, Gres, Hres)
        # Evaluate model-improving points to pick best one
        # May eventually want to normalize Mdir first for infty norm
        # Plus directions
        [Mdir1, mp1] = bmpts(X[xk_in], Mdir[0 : n - mp, :], Low, Upp, delta, Model["Par"][2])
        for i in range(n - mp1):
            D = Mdir1[i, :]
            Res[i, 0] = D @ (G + 0.5 * H @ D.T)
        b = np.argmin(Res[: n - mp1, 0:1])
        a1 = np.min(Res[: n - mp1, 0:1])
        Xsp = Mdir1[b, :]
        # Minus directions
        [Mdir1, mp2] = bmpts(X[xk_in], -Mdir[0 : n - mp, :], Low, Upp, delta, Model["Par"][2])
        for i in range(n - mp2):
            D = Mdir1[i, :]
            Res[i, 0] = D @ (G + 0.5 * H @ D.T)
        b = np.argmin(Res[: n - mp2, 0:1])
        a2 = np.min(Res[: n - mp2, 0:1])
        if a2 < a1:
            Xsp = Mdir1[b, :]
        nf += 1
        X[nf] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Xsp))  # Temp safeguard
        F[nf] = Ffun(X[nf])
        if np.any(np.isnan(F[nf])):
            X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
            return X, F, hF, flag, xk_in
        hF[nf] = hfun(F[nf])
        if printf:
            print("%4i   Model point     %11.5e\n" % (nf, hF[nf]))
        if hF[nf] < hF[xk_in]:  # ! Eventually check stuff decrease here
            if printf:
                print("**improvement from model point****")
            # Update model to reflect new base point
            D = X[nf] - X[xk_in]
            xk_in = nf  # Change current center
            Cres = F[xk_in]
            # Don't actually use
            # for j in range(m):
            #     Gres[:, j] = Gres[:, j] + Hres[:, :, j] @ D.T

    return Cres, Gres, Hres, nf, X, F, hF, valid, xk_in, mp
