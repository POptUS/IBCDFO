import numpy as np
from scipy.optimize import linprog
from cvxopt import matrix, solvers
import cvxpy as cp
import ipdb

def vech(A):
    return A[np.tril_indices_from(A)]

def vech_inv(v):
    # Determine size n of the matrix from the length of v = n(n+1)/2
    l = len(v)
    n = int((np.sqrt(8 * l + 1) - 1) / 2)
    
    if n * (n + 1) // 2 != l:
        raise ValueError("Length of input vector is not consistent with any square matrix vech.")
    
    A = np.zeros((n, n))
    tril_indices = np.tril_indices(n)
    A[tril_indices] = v
    A = A + A.T - np.diag(np.diag(A))  # Make symmetric
    return A

def duplication_matrix(n): # vec(A) = D @ vech(A)
    p = n * (n + 1) // 2  # length of vech(A)
    D = np.zeros((n * n, p))

    count = 0
    for j in range(n):
        for i in range(j, n):
            e_ij = np.zeros((n, n))
            e_ij[i, j] = 1
            if i != j:
                e_ij[j, i] = 1

            D[:, count] = e_ij.flatten(order='F')  # column-major flatten (vec)
            count += 1

    return D

import numpy as np

def elimination_matrix(n): # vech(A) = L @ vec(A)
    p = n * (n + 1) // 2
    L = np.zeros((p, n * n))

    count = 0
    for j in range(n):
        for i in range(j, n):
            index = i + j * n  # column-major index (vec)
            L[count, index] = 1
            if i != j:
                index_symmetric = j + i * n
                L[count, index_symmetric] = 1
                L[count, index] = 0.5
                L[count, index_symmetric] = 0.5
            count += 1

    return L

def add_row_if_increases_rank(A, b, new_row, new_ele):
    """
    Adds new_row to matrix A if it increases the rank of A.
    
    Args:
        A: numpy array of shape (m, n)
        new_row: numpy array of shape (n,)
    
    Returns:
        A_updated: numpy array with the new row added if it increases rank
    """
    if A.size == 0:
        rank_before = 0
    else:
        rank_before = np.linalg.matrix_rank(A, tol=1e-8)
    A_new = np.vstack([A, new_row])
    b_new = np.append(b, new_ele)
    rank_after = np.linalg.matrix_rank(A_new, tol=1e-8)

    if rank_after > rank_before:
        return A_new, b_new
    else:
        return A, b  # return unchanged matrix

def min_norm_interpolation_hessian(X, nf, xkin, F, G_k, f_k, beta, h, Xlist):

    Xlist = np.array(Xlist).astype(int)

    # Inputs:
    x_k = X[xkin]  # current point x_k
    n = x_k.shape[0]  # dimension

    Dn = duplication_matrix(n)  # duplication matrix
    P = Dn.T @ Dn  # P = D^T D
    # P = 0.5 * (P + P.T)

    # A = np.zeros((len(Xlist), n * (n + 1) // 2))
    # b = np.zeros(len(Xlist))
    A = np.empty((0, n * (n + 1) // 2))
    b = np.empty((0,))
    for i, index in enumerate(Xlist):
        x_diff = X[index] - x_k
        # print("x_diff:", x_diff)
        S = np.outer(x_diff, x_diff)
        linear_term = max(np.dot(G_k.T, x_diff))  # Assuming G_k is a matrix of gradients
        # m_ki = linear_term + f_k[i] - beta[i]
        m_ki = linear_term + f_k[i]
        new_row = vech(S)
        new_ele = h[index] - m_ki
        A, b = add_row_if_increases_rank(A, b, new_row, new_ele)
        # b = np.append(b, h[index] - m_ki)

    # print("b:", b)

    if A.size == 0:
        return np.zeros((n, n))  # If no constraints, return zero matrix
    

    A = A @ P
    # print("A:", A)
    # print("b:", b)

    P  = matrix(P)  # P = D^T D
    q  = matrix(np.zeros((n * (n + 1) // 2, 1)))  # q = 0


    A = matrix(A)
    b = matrix(b)
    solution = solvers.qp(P, q, None, None, A, b)
    h_opt = np.array(solution['x']).flatten()
    # print("Optimal h_opt:", h_opt)
    H = vech_inv(h_opt)

    print("Optimal H:", H)

    # Define symmetric variable H
    # H = cp.Variable((n, n), symmetric=True)

    # print("Xlist:", Xlist)
    # print("n:", n)

    # # Define constraints
    # constraints = []
    # for i, index in enumerate(Xlist):
    #     x_diff = X[index] - x_k
    #     linear_term = max(np.dot(G_k.T, x_diff))  # Assuming G_k is a matrix of gradients
    #     quad_term = 0.5 * cp.quad_form(x_diff, H)
    #     m_ki = h[xkin] + linear_term - beta[i] + quad_term
    #     constraints.append(m_ki == h[Xlist[i]])

    # # Define problem
    # objective = cp.Minimize(cp.norm(H, 2))  # Spectral norm
    # prob = cp.Problem(objective, constraints)

    # # Solve it
    # prob.solve()
    # print("Problem status:", prob.status)

    # # Output the optimal H
    # print("Optimal H:", H.value)
    return H
