import numpy as np

n = 10  # Number of matrices to sum together
m = 20  # Dimension of the square matrices

np.random.seed(0)
A = {}
for i in range(n):
    # Make 5 random positive-and negative-definite matricies
    B = np.random.uniform(-1, 1, (m, m))
    A[i] = (-1) ** i * 0.5 * (B @ B.T)


def compute_M_and_eig(y):
    y = y.squeeze()
    assert len(y) == n, "Wrong input dimension"
    M = np.zeros((m, m))
    for i, val in enumerate(y):
        M += val * A[i]

    # eigvals, eigvecs = np.linalg.eigh(M)
    eigvals, eigvecs = np.linalg.eig(M)
    return eigvals, eigvecs


def Ffun_default(y):
    eigvals, _ = compute_M_and_eig(y)

    return eigvals


def Ffun_sort(y):
    eigvals, _ = compute_M_and_eig(y)
    eigvals.sort()

    return eigvals
