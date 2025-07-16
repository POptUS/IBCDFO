import numpy as np
from numpy.linalg import eigh
from scipy.linalg import solve_lyapunov
from pymanopt.manifolds.manifold import Manifold
import ipdb

def sqrt_psd(A, tol):
    A = (A + A.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals_clipped = np.clip(eigvals, a_min=tol, a_max=np.inf)
    sqrt_diag = np.sqrt(eigvals_clipped)
    return eigvecs @ np.diag(sqrt_diag) @ eigvecs.conj().T


class DensityMatrixManifold(Manifold):
    def __init__(self, n, tol=1e-6):
        self._n = n
        self._name = f"Density matrices of size {n}x{n}"
        self._tol = tol

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        # Hermitian traceless matrices: n^2 - 1 (real degrees of freedom)
        return self._n**2 - 1

    @property
    def point_layout(self):
        return 1

    def inner_product(self, rho, X, Y):
        # Quantum Fisher inner product on the support of rho
        evals, evecs = eigh(rho)
        support = evals > self._tol
        Lambda = np.diag(evals[support])
        U = evecs[:, support]

        X_tilde = U.conj().T @ X @ U
        Y_tilde = U.conj().T @ Y @ U

        LX_inv_Y = -solve_lyapunov(Lambda, -Y_tilde)
        LX_inv_X = -solve_lyapunov(Lambda, -X_tilde)

        return 0.5 * np.real(np.trace(X_tilde @ LX_inv_Y + Y_tilde @ LX_inv_X))

    def norm(self, rho, X):
        return np.sqrt(self.inner_product(rho, X, X))

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return self.projection(point, point @ euclidean_gradient @ point)

    def dist(self, rho, sigma, tol=1e-6):
        rho = (rho + rho.conj().T) / 2
        sigma = (sigma + sigma.conj().T) / 2

        sqrt_rho = sqrt_psd(rho, tol)
        middle = sqrt_rho @ sigma @ sqrt_rho
        middle = (middle + middle.conj().T) / 2

        sqrt_middle = sqrt_psd(middle, tol)

        fidelity = np.real(np.trace(sqrt_middle)) ** 2
        fidelity = np.clip(fidelity, 0.0, 1.0)

        return np.sqrt(2.0 * (1.0 - np.sqrt(fidelity)))

    def projection(self, rho, A):
        # Project A onto the tangent space at rho (traceless, Hermitian, support)
        evals, evecs = eigh(rho)
        support = evals > self._tol
        U = evecs[:, support]
        Lambda = np.diag(evals[support])
        r = Lambda.shape[0]

        A_tilde = U.conj().T @ A @ U
        A0_tilde = A_tilde - (np.trace(A_tilde) / r) * np.eye(r)

        L_A0_tilde = Lambda @ A0_tilde + A0_tilde @ Lambda
        lambda_scalar = (2 / r) * np.trace(Lambda @ A0_tilde)
        RHS = L_A0_tilde - lambda_scalar * np.eye(r)

        X_tilde = solve_lyapunov(Lambda, -RHS)
        X_proj = U @ X_tilde @ U.conj().T
        return (X_proj + X_proj.conj().T) / 2  # Hermitian

    def random_point(self):
        # Generate a random full-rank density matrix
        A = np.random.randn(self._n, self._n) + 1j * np.random.randn(self._n, self._n)
        A = (A + A.conj().T) / 2
        eigvals, eigvecs = eigh(A)
        eigvals = np.exp(eigvals)  # Make PSD
        eigvals /= np.sum(eigvals)  # Normalize trace to 1
        return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

    def random_tangent_vector(self, rho):
        # Generate a random Hermitian traceless matrix in support
        X = np.random.randn(self._n, self._n) + 1j * np.random.randn(self._n, self._n)
        X = (X + X.conj().T) / 2
        X -= np.trace(X) / self._n * np.eye(self._n)
        return self.projection(rho, X)


    def retraction(self, rho, X, tol=1e-6):

        # First-order retraction: rho + X, projected back to trace 1 and PSD
        # Step 1: Take a step in ambient space
        Y = rho + X
        Y = (Y + Y.conj().T) / 2  # enforce Hermitian

        # Step 2: Project to PSD cone via eigendecomposition
        evals, evecs = np.linalg.eigh(Y)
        evals_clipped = np.clip(evals, 0.0, None)
        Y_proj = evecs @ np.diag(evals_clipped) @ evecs.conj().T

        # Step 3: Normalize to trace one
        trace_Y = np.trace(Y_proj)
        if trace_Y <= tol:
            raise ValueError("Retracted matrix is numerically zero.")
        rho_new = Y_proj / trace_Y
        rho_new = (rho_new + rho_new.conj().T) / 2  # final symmetrize

        return rho_new


    def zero_vector(self, rho):
        return np.zeros_like(rho)

    def exp(self, rho, X, tol=1e-6):
        """
        Exponential map on the manifold of full-rank density matrices under the Bures metric.
        Uses eigendecomposition instead of sqrtm or inv.
        """
        # Step 1: Eigendecomposition of rho
        evals, U = np.linalg.eigh((rho + rho.conj().T) / 2)
        evals = np.clip(evals, tol, None)
        sqrt_Lambda = np.sqrt(evals)
        inv_sqrt_Lambda = 1.0 / sqrt_Lambda

        sqrt_rho = U @ np.diag(sqrt_Lambda) @ U.conj().T
        sqrt_rho_inv = U @ np.diag(inv_sqrt_Lambda) @ U.conj().T

        # Step 2: Define W = sqrt(I + sqrt_rho^{-1} X sqrt_rho^{-1})
        A = sqrt_rho_inv @ X @ sqrt_rho_inv
        A = (A + A.conj().T) / 2
        # For small X, this approximates a geodesic step: use expm or sqrtm if needed
        W = np.eye(rho.shape[0]) + 0.5 * A  # 1st-order approximation
        rho_new = sqrt_rho @ W @ W.conj().T @ sqrt_rho
        rho_new = (rho_new + rho_new.conj().T) / 2
        rho_new /= np.trace(rho_new)

        return rho_new

    def log(self, rho, sigma, tol=1e-6):
        """
        Logarithmic map from rho to sigma under Bures metric using eigendecomposition.
        Returns a Hermitian, traceless tangent vector at rho.
        """
        # Eigendecomposition of rho
        evals_rho, U = np.linalg.eigh((rho + rho.conj().T) / 2)
        evals_rho = np.clip(evals_rho, tol, None)
        sqrt_rho = U @ np.diag(np.sqrt(evals_rho)) @ U.conj().T
        sqrt_rho_inv = U @ np.diag(1.0 / np.sqrt(evals_rho)) @ U.conj().T

        # Compute middle matrix: M = sqrt( sqrt(rho) * sigma * sqrt(rho) )
        middle = sqrt_rho @ sigma @ sqrt_rho
        middle = (middle + middle.conj().T) / 2
        eigvals_m, eigvecs_m = np.linalg.eigh(middle)
        eigvals_m = np.clip(eigvals_m, tol, None)
        sqrt_middle = eigvecs_m @ np.diag(np.sqrt(eigvals_m)) @ eigvecs_m.conj().T

        # W = sqrt_rho^{-1} * sqrt_middle * sqrt_rho^{-1}
        W = sqrt_rho_inv @ sqrt_middle @ sqrt_rho_inv
        log_term = W @ W - np.eye(rho.shape[0])

        X = sqrt_rho @ log_term @ sqrt_rho
        X = (X + X.conj().T) / 2
        X -= np.trace(X) / rho.shape[0] * np.eye(rho.shape[0])  # enforce traceless

        return X

    def pair_mean(self, rho, sigma):
        """
        Compute the intrinsic mean (Riemannian midpoint) between two density matrices
        under the Bures metric. Handles rank-deficient inputs via support projection.
        """
        rho = (rho + rho.conj().T) / 2
        sigma = (sigma + sigma.conj().T) / 2

        evals, evecs = np.linalg.eigh(rho)
        support_mask = evals > self._tol
        if not np.any(support_mask):
            raise ValueError("rho is numerically zero; undefined mean.")

        # Extract support subspace
        U = evecs[:, support_mask]
        evals_supported = evals[support_mask]
        sqrt_rho_support = U @ np.diag(np.sqrt(evals_supported)) @ U.conj().T

        # Project sigma into support
        sigma_support = U.conj().T @ sigma @ U

        # Compute W in support basis
        temp = sqrt_rho_support @ sigma @ sqrt_rho_support
        temp = (temp + temp.conj().T) / 2
        evals_temp, vecs_temp = np.linalg.eigh(temp)
        sqrt_temp = vecs_temp @ np.diag(np.sqrt(np.clip(evals_temp, self._tol, None))) @ vecs_temp.conj().T

        # Compute mean in support
        midpoint_support = (sqrt_rho_support + sqrt_temp) / 2
        rho_mean = midpoint_support @ midpoint_support
        rho_mean = (rho_mean + rho_mean.conj().T) / 2

        # Normalize trace
        rho_mean /= np.trace(rho_mean)
        return rho_mean

    def transport(self, rho, sigma, X):
        """
        Vector transport of X ∈ T_rho M to T_sigma M under the Bures metric.
        Handles rank-deficient rho via support-restricted subspace transport.
        """
        rho = (rho + rho.conj().T) / 2
        sigma = (sigma + sigma.conj().T) / 2
        X = (X + X.conj().T) / 2

        # Step 1: Eigenbasis of rho
        evals_rho, U_rho = np.linalg.eigh(rho)
        support_mask = evals_rho > self._tol
        if not np.any(support_mask):
            raise ValueError("rho is numerically zero.")

        U = U_rho[:, support_mask]  # Support basis
        Lambda = evals_rho[support_mask]
        sqrt_Lambda = np.sqrt(Lambda)
        inv_sqrt_Lambda = 1.0 / sqrt_Lambda

        # Step 2: Project everything into support of rho
        X_tilde = U.conj().T @ X @ U
        sigma_tilde = U.conj().T @ sigma @ U

        # Step 3: sqrt(sigma_tilde) in support
        evals_sigma, V_sigma = np.linalg.eigh(sigma_tilde)
        evals_sigma = np.clip(evals_sigma, self._tol, None)
        sqrt_sigma_tilde = V_sigma @ np.diag(np.sqrt(evals_sigma)) @ V_sigma.conj().T

        # Step 4: Compute transport matrix A = sqrt_sigma @ inv(sqrt_rho)
        A = sqrt_sigma_tilde @ np.diag(inv_sqrt_Lambda)

        # Step 5: Transport: X ↦ A X A
        X_new_tilde = A @ X_tilde @ A.conj().T
        X_new_tilde = (X_new_tilde + X_new_tilde.conj().T) / 2

        # Step 6: Return transported X in ambient space
        X_new = U @ X_new_tilde @ U.conj().T
        X_new = (X_new + X_new.conj().T) / 2
        X_new -= np.trace(X_new) / self._n * np.eye(self._n)  # Ensure traceless

        return X_new


