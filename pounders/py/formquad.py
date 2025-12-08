import numpy as np
import scipy.linalg

from .phi2eval import phi2eval

# from .flipFirstRow import flipFirstRow
# from .flipSignQ import flipSignQ


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
    Pars[4] [log] Flag to find affine points in forward order (0)
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
    scale_mat = np.ones((n, n)) / np.sqrt(2)
    scale_mat[np.diag_indices(n)] = 1
    inds_to_use_in_H = np.triu_indices(n)

    assert isinstance(np_max, (int, np.integer)), "np_max must be an integer"
    assert isinstance(xk_in, (int, np.integer)), "xk_in must be an integer"

    D = (X[:nf] - X[xk_in]) / delta
    D = D.astype(np.float64)  # Convert entries to float to use qr_insert
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
        # Order to look for Affinely independent points (generators in Python can't be rewound)
        if not Pars[4]:
            indorder = reversed(range(nf))
        else:
            indorder = range(nf)
        for i in indorder:
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

    # Precompute L^T L and its Cholesky factorization once
    LTL = L.T @ L
    LTL_cf, LTL_lower = scipy.linalg.cho_factor(LTL, lower=False)

    # Precompute helpers for the M / Alpha solve
    M_rows, M_cols = M.shape
    is_square_M = M_rows == M_cols

    use_MT = None
    MT_pinv = None
    M_pinv = None

    if not is_square_M:
        # Decide which least-squares system you used in the original code
        if M_rows != N.shape[1]:
            # Original code used np.linalg.lstsq(M.T, ...)
            MT_pinv = np.linalg.pinv(M.T)  # (M.T)^+ precomputed once
            use_MT = True
        else:
            # Original code used np.linalg.lstsq(M, ...)
            M_pinv = np.linalg.pinv(M)  # M^+ precomputed once
            use_MT = False

    # ---- Vectorized replacement for the loop over m ----

    if L.shape[0] != N.shape[0]:
        # Zero-Hessian case: set all quadratic coefficients to zero directly.
        num_beta = N.shape[0]  # == number of entries in upper triangle
        Beta_all = np.zeros((num_beta, m))  # shape: (num_beta, m)
        rhs_all = F  # since N.T @ Beta_all == 0 (N is zero)
    else:
        # Normal case: solve for Omega_all from (L^T L) * Omega_all = Z^T * F
        J_all = Z.T @ F  # shape: (p, m), p = number of columns of Z

        Omega_all = scipy.linalg.cho_solve(
            (LTL_cf, LTL_lower),
            J_all,  # same Cholesky factor, multiple RHS
        )  # shape: (p, m)

        # Quadratic coefficients (packed Hessian entries) for all k
        Beta_all = L @ Omega_all  # shape: (num_beta, m)

        # Right-hand side for all k
        rhs_all = F - N.T @ Beta_all  # shape: (mp, m)

    # Solve for Alpha for all k
    if is_square_M:
        # np.linalg.solve supports multiple RHS as columns
        Alpha_all = np.linalg.solve(M.T, rhs_all)  # shape: (#alpha, m)
    else:
        if use_MT:
            Alpha_all = MT_pinv @ rhs_all  # shape: (#alpha, m)
        else:
            Alpha_all = M_pinv @ rhs_all  # shape: (#alpha, m)

    # Fill G for all k (gradients)
    G[:, :] = Alpha_all[1 : n + 1, :]  # same slice as in the loop

    # Fill H for all k using the packed Beta_all and the mask.
    # H has shape (n, n, m); inds_to_use_in_H is a boolean mask on (n, n).
    # H[inds_to_use_in_H] has shape (num_triu, m), matching Beta_all.
    H[inds_to_use_in_H] = Beta_all  # upper triangle
    H.transpose(1, 0, 2)[inds_to_use_in_H] = Beta_all  # lower triangle (mirror)

    # Apply scaling to all k
    H *= scale_mat[..., None]

    H = H / (delta**2)
    G = G / delta

    return [Mdir, mp, valid, G, H, Mind]
