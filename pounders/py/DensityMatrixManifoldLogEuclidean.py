import numpy as np
from pymanopt.manifolds.manifold import Manifold
from scipy.linalg import logm, expm, sqrtm

class DensityMatrixManifold(Manifold):
    def __init__(self, n, tol=1e-12):
        self._n = n
        self._name = f"Density matrices (PSD Hermitian, trace 1), n = {n}"
        self._tol = tol
    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        # Hermitian traceless matrices: n^2 - 1 (real degrees of freedom)
        return self._n ** 2 - 1

    @property
    def point_layout(self):
        return 1

    def inner_product(self, X, G, H):
        # Log-Euclidean inner product: ⟨A, B⟩_X = ⟨log(X)^{1/2} A log(X)^{1/2}, log(X)^{1/2} B log(X)^{1/2}⟩_F
        logX = logm(X + 1e-3 * np.eye(self._n))  # Regularization to avoid log(0)
        return np.real(np.trace(logX @ G @ logX @ H))

    def norm(self, X, G):
        return np.sqrt(self.inner_product(X, G, G))

    def projection(self, X, H):
        # Project H onto the tangent space: Hermitian + trace zero
        H_herm = 0.5 * (H + H.conj().T)
        return H_herm - np.trace(H_herm) * np.eye(self._n) / self._n

    def random_point(self):
        A = np.random.randn(self._n, self._n) + 1j * np.random.randn(self._n, self._n)
        A = (A + A.conj().T) / 2
        A = A @ A.conj().T
        A /= np.trace(A)
        return A

    #def randvec(self, X):
    #    H = np.random.randn(self._n, self._n) + 1j * np.random.randn(self._n, self._n)
    #    H = self.proj(X, H)
    #    return H / self.norm(X, H)

    def zero_vector(self, X):
        return np.zeros_like(X)

    def retraction(self, X, G):
        # Use exponential retraction, then normalize trace
        Y = X + G
        Y = 0.5 * (Y + Y.conj().T)  # Ensure Hermitian
        eigvals, eigvecs = np.linalg.eigh(Y)
        eigvals = np.maximum(eigvals, 0)  # Project to PSD
        Y = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        Y /= np.trace(Y)
        return Y

    def euclidean_to_riemannian_gradient(self, X, G):
        # Euclidean gradient to Riemannian gradient: projection onto tangent space
        return self.projection(X, G)

    def dist(self, X, Y):
        # Regularize both matrices slightly to ensure numerical stability
        eps = 1e-3
        X_reg = X + eps * np.eye(self._n)
        Y_reg = Y + eps * np.eye(self._n)

        logX = logm(X_reg)
        logY = logm(Y_reg)

        diff = logX - logY
        return np.linalg.norm(diff, 'fro')

    def random_tangent_vector(self, rho):
        # Generate a random Hermitian traceless matrix in support
        X = np.random.randn(self._n, self._n) + 1j * np.random.randn(self._n, self._n)
        X = (X + X.conj().T) / 2
        X -= np.trace(X) / self._n * np.eye(self._n)
        return self.proj(rho, X)

    def transport(self, X, Y, eta):
        # In log-Euclidean geometry, transport is just projecting the vector to the new tangent space
        return self.projection(Y, eta)
