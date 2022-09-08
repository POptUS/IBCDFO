import numpy as np


import scipy.linalg


from phi2eval import phi2eval


from flipFirstRow import flipFirstRow


from flipSignQ import flipSignQ


def formquad(X, F, delta, xkin, mpmax, Pars, vf):
    '''
    formquad(X,F,delta,xkin,npmax,Pars,vf) -> [Mdir,np,valid,G,H,Mind]
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
    xkin    [int] Index in (X and F) of the current center
    npmax   [int] Max # interpolation points (>n+1) (.5*(n+1)*(n+2))
    Pars[0] [dbl] delta multiplier for checking validity
    Pars[1] [dbl] delta multiplier for all interpolation points
    Pars[2] [dbl] Pivot threshold for validity
    Pars[3] [dbl] Pivot threshold for additional points (.001)
    vf      [log] Flag indicating you just want to check model validity
    --OUTPUTS----------------------------------------------------------------
    Mdir    [dbl] [(n-np+1)-by-n]  Unit directions to improve model
    np      [int] Number of interpolation points (=length(Mind))
    valid   [log] Flag saying if model is valid within Pars[2]*delta
    G       [dbl] [n-by-m]  Matrix of model gradients at Xk
    H       [dbl] [n-by-n-by-m]  Array of model Hessians at Xk
    Mind    [int] [npmax-by-1] Integer vector of model interpolation indices
    '''
    # % --DEPENDS ON-------------------------------------------------------------
    # phi2eval : Evaluates the quadratic basis for vector inputs
    # qrinsert, svd : scipy.linalg and numpy
    # Internal parameters:
    nf, n = np.shape(X)
    m = np.shape(F)[1]
    G = np.zeros((n, m))
    H = np.zeros((n, n, m))
    # Precompute the scaled displacements (could be expensive for larger nfmax)
    D = np.zeros((nf, n))  # Scaled displacements
    Nd = np.zeros((nf, 1))  # Norm of scaled displacements
    if type(mpmax) != int:
        mpmax = mpmax[0, 0]
    if type(xkin) != int:
        xkin = xkin[0, 0]  # If xkin is a numpy array 1 x 1, cast it as an int

    D = (X[:nf] - X[xkin])/delta
    Nd = np.atleast_2d(np.linalg.norm(D,2,axis=1)).T

    # Get n+1 sufficiently affinely independent points:
    # Initialize the QR factorization of interest
    Q = np.eye(n)
    R = np.empty(shape=(0, 0))
    # Indices of model interpolation points
    Mind = np.array([[xkin]])
    valid = False
    # Counter for number of interpolation points
    mp = 0
    for aff in range(0, 2):
        for i in reversed(range(0, nf)):
            if Nd[i, 0] <= Pars[aff]:
                proj = np.linalg.norm(D[i:i+1, :] @ Q[:, mp: n], 2)  # Project D onto null
                if proj >= Pars[aff + 2]:  # add this index to Mind
                    mp += 1
                    Mind = np.vstack((Mind, np.array([[i]])))
                    if np.shape(R)[0] == 0:
                        [Q, R] = np.linalg.qr(D[i:i+1, :].T, mode='complete')
                        # [Q, R] = flipFirstRow(Q, R, 0, np.shape(Q)[1]-1)
                        # [Q, R] = flipSignQ(Q, R, 0, np.shape(Q)[1]-1)
                    else:
                        # Update QR
                        D[i:i+1, :] = np.float64(D[i:i+1, :])  # Convert entries to float to use qr_insert
                        [Q, R] = scipy.linalg.qr_insert(Q, R, D[i:i+1, :].T, mp - 1, 'col')
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
    N = phi2eval(D[Mind.flatten(), :]).T

    mp = len(Mind) 
    M = np.hstack((np.ones((n+1, 1)), D[Mind.T[0], :])).T
    [Q, R] = np.linalg.qr(M.T, mode='complete')
    # [Q, R] = flipSignQ(Q, R, 0, np.shape(Q)[1]-1)
    # Now we add points until we have mpmax starting with the most recent ones
    i = nf - 1
    while mp < mpmax or mpmax == n+1:
        if Nd[i, 0] <= Pars[1] and i not in Mind:
            Ny = np.hstack((N, phi2eval(D[i:i+1, :]).T))
            # Update QR
            D[i, :] = np.float64(D[i, :])  # Convert entries to float to use qr_insert
            [Qy, Ry] = scipy.linalg.qr_insert(Q, R, np.hstack((np.array([[1.]]), D[i:i+1, :])), mp, 'row')
            # [Qy, Ry] = flipFirstRow(Qy, Ry, 0, np.shape(Q)[1]-1)
            # [Qy, Ry] = flipSignQ(Qy, Ry, 0, np.shape(Q)[1]-1)
            Ly = Ny @ Qy[:, n+1: mp + 1]
            _, s, _ = np.linalg.svd(Ly)
            if min(s) > Pars[3]:
                mp += 1
                Mind = np.vstack((Mind, [[i]]))
                N = Ny
                Q = Qy
                R = Ry
                L = Ly
                Z = Q[:, n+1: mp]
                # Note that M is growing
                M = np.hstack((M, np.vstack((np.array([[1]]), D[i:i+1, :].T))))
        i -= 1
        # Reached end of points
        if i == -1:
            # Set outputs so that Hessian is zero
            if mp == n + 1:
                # L = 1
                L = np.array([[1]])
                Z = np.zeros((n+1, int(0.5 * n * (n+1))))
                N = np.zeros((int(0.5 * n * (n+1)), n+1))
            break
    F = F[Mind.T[0], :]
    for k in range(0, m):
        # For L = N * Z, solve L.T * L * Omega = Z.T * f:
        if np.shape(L)[1] != np.shape(Z.T @ F[:, k:k+1])[1]:
            Omega = np.linalg.solve(L.T @ L, (Z.T @ F[:, k:k+1]))
        else:
            Omega = np.linalg.solve(L.T @ L, (Z.T @ F[:, k:k+1]).T)
        Beta = L @ Omega
        if np.shape(Beta)[1] > 1:
            Beta = Beta.T
        if np.shape(M)[0] == np.shape(M)[1]:
            Alpha = np.linalg.solve(M.T, F[:, k:k+1] - N.T @ Beta)
        else:
            if np.shape(M)[0] != np.shape(N)[1]:
                Alpha = np.linalg.lstsq(M.T, F[:, k:k+1] - N.T @ Beta, rcond=None)[0]
            else:
                Alpha = np.linalg.lstsq(M, F[:, k:k+1] - N.T @ Beta, rcond=None)[0]
            Alpha = np.reshape(Alpha, (np.shape(Alpha)[0], 1))
        G[:, k] = Alpha[1:n+1, 0]
        num = -1
        for i in range(0, n):
            num += 1
            H[i, i, k] = Beta[num, 0]
            for j in range(i+1, n):
                num += 1
                H[i, j, k] = Beta[num, 0] / np.sqrt(2)
                H[j, i, k] = H[i, j, k]
    H = H / (delta ** 2)
    G = G / delta

    Mind = np.squeeze(Mind)
    return [Mdir, mp, valid, G, H, Mind]
