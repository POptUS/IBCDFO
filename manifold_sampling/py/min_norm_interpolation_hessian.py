import numpy as np
from scipy.optimize import linprog
from cvxopt import matrix, solvers
import cvxpy as cp
import ipdb
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting


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
    rank_after = np.linalg.matrix_rank(A_new, tol=1e-8)

    if rank_after > rank_before:
        return True
    else:
        return False  # return unchanged matrix

def linear_model_k(s, b, G):
    # mk_s = max(np.expand_dims(bk, 1) + np.dot(G_k.T, s))
    lineark_s = max(b + np.dot(G.T, s))
    return lineark_s

def model_k(s, b, G, H):

    mk_s = linear_model_k(s, b, G) + 0.5 * np.dot(s, np.dot(H, s))
    return mk_s
    

# def min_norm_interpolation_hessian(X, nf, xkin, F, G_k, f_k, beta, h, Xlist):
def min_norm_interpolation_hessian(X, xkin, f, f_bar, beta, G_k, h, Xlist, hfun, Ffun, delta, Hash, plot_m=False):

    Xlist = np.array(Xlist).astype(int)

    x_k = X[xkin]  # current point x_k
    n = x_k.shape[0]  # dimension

    Dn = duplication_matrix(n)  # duplication matrix
    P = Dn.T @ Dn  # P = D^T D

    bk = f_bar - beta # - f
    A = np.empty((0, n * (n + 1) // 2))
    b = np.empty((0,))
    added_rows = np.empty((0,))

    # XkDist = cdist(X[Xlist], X[xkin: xkin+1], metric="chebyshev").flatten()
    # # print("XkDist:", XkDist)
    # sorted_dis = np.argsort(XkDist)
    # sorted_idx = Xlist[sorted_dis]
    measure = np.array([np.abs(hfun(Ffun(X[idx])[0])[0] - linear_model_k(X[idx] - x_k, bk, G_k))
                    for idx in Xlist])
    sorted_dis = np.argsort(measure)[::-1]
    sorted_idx = Xlist[sorted_dis]
    for i, index in enumerate(sorted_idx):
        x_diff = X[index] - x_k
        S = np.outer(x_diff, x_diff)
        linear_i = linear_model_k(x_diff, bk, G_k)
        new_row = vech(S)
        new_ele = h[index] - linear_i
        add_flag = add_row_if_increases_rank(A, b, new_row, new_ele)

        if add_flag == True:
            if added_rows.size == 0:
                add_flag = True
            else:
                for j in added_rows:
                    if Hash[index] == Hash[j]:
                        add_flag = False
                        break
        if add_flag == True:
            print(Hash[index])
            added_rows = np.append(added_rows, index)
            A = np.vstack([A, new_row])
            b = np.append(b, new_ele)

    added_rows = np.array(added_rows).astype(int)

    if A.size == 0:
        return np.zeros((n, n))  # If no constraints, return zero matrix
    
    b = b*2

    A = A @ P

    P  = matrix(P)  # P = D^T D
    q  = matrix(np.zeros((n * (n + 1) // 2, 1)))  # q = 0


    A = matrix(A)
    b = matrix(b)
    solution = solvers.qp(P, q, None, None, A, b)
    h_opt = np.array(solution['x']).flatten()
    H = vech_inv(h_opt)


    # measure_after = np.array([np.abs(hfun(Ffun(X[idx])[0])[0]- model_k(X[idx] - x_k, bk, G_k, H)) for idx in Xlist])
    # # print(np.linalg.norm(measure-measure_after, np.inf))
    # sorted_dis_after = np.argsort(measure_after)
    # sorted_idx_after = Xlist[sorted_dis_after]
    # # print("Measure on the interpolation points:", measure_after)

    # A = np.empty((0, n * (n + 1) // 2))
    # b = np.empty((0,))
    # added_rows = np.empty((0,))


    # for i, index in enumerate(sorted_idx_after):
    #     # x_diff = X[Xlist[index]] - x_k
    #     x_diff = X[index] - x_k
    #     # print("x_diff:", x_diff)
    #     S = np.outer(x_diff, x_diff)
    #     linear_i = linear_model_k(x_diff, bk, G_k)
    #     new_row = vech(S)
    #     new_ele = h[index] - linear_i
    #     A, b, add_flag = add_row_if_increases_rank(A, b, new_row, new_ele)

    #     if add_flag == True:
    #         added_rows = np.append(added_rows, index)

    # added_rows = np.array(added_rows).astype(int)

    # if A.size == 0:
    #     return np.zeros((n, n))  # If no constraints, return zero matrix
    
    # b = b*2

    # A = A @ P

    # P  = matrix(P)  # P = D^T D
    # q  = matrix(np.zeros((n * (n + 1) // 2, 1)))  # q = 0


    # A = matrix(A)
    # b = matrix(b)
    # solution = solvers.qp(P, q, None, None, A, b)
    # h_opt = np.array(solution['x']).flatten()
    # # print("Optimal h_opt:", h_opt)
    # H = vech_inv(h_opt)
    # measure_after = np.array([np.abs(hfun(Ffun(X[idx])[0])[0]- model_k(X[idx] - x_k, bk, G_k, H)) for idx in Xlist])
    # print("Measure on the interpolation points after:", measure_after)


    if plot_m == True:

        x = np.linspace(x_k[0]-1, x_k[0]+1, 100)
        y = np.linspace(x_k[1]-1, x_k[1]+1, 100)
        X_mesh, Y_mesh = np.meshgrid(x, y)

        # Apply the function to each point in the grid
        Z1 = np.array([[model_k([x, y]-x_k, bk, G_k, H) for x, y in zip(row_x, row_y)] 
                    for row_x, row_y in zip(X_mesh, Y_mesh)])
        

        # Apply the function to each point in the grid
        Z2 = np.array([[hfun(Ffun(np.array([x, y]))[0])[0] for x, y in zip(row_x, row_y)] 
                    for row_x, row_y in zip(X_mesh, Y_mesh)])
        
        # ax = fig.add_subplot(111, projection='3d')

        contour1 = plt.contour(X_mesh, Y_mesh, Z1, cmap='viridis', alpha=0.7, levels=40, linestyles='dashed')
        contour2 = plt.contour(X_mesh, Y_mesh, Z2, cmap='viridis', alpha=0.7, levels=40)
        plt.clabel(contour1, inline=True, fontsize=10)
        plt.clabel(contour2, inline=True, fontsize=10)
        plt.colorbar(contour1, label='m_k')
        plt.colorbar(contour2, label='f')

        plt.scatter(x_k[0], x_k[1], marker='o', color='red', s=10, label='x_k')  # Current point


        error = []
        for i, index in enumerate(added_rows):
            x_ind = X[index]
            if index in added_rows:
                marker = 'x'
            else:
                marker = 's'
            plt.scatter(x_ind[0], x_ind[1], marker=marker, color='blue', s=10, label=f'X[{index}]')
            error.append(np.abs(hfun(Ffun(x_ind)[0])[0] - model_k(x_ind - x_k, bk, G_k, H)))
            # print("error", np.abs(hfun(Ffun(x_ind)[0])[0] - model_k(x_ind - x_k, bk, G_k, H)))
        print("error on the interpolation points:", max(error))

        # plt.contour(X_mesh, Y_mesh, Z1, levels=20, cmap='viridis', linestyles='solid')
        # plt.contour(X_mesh, Y_mesh, Z2, levels=20, cmap='plasma', linestyles='dashed')
        plt.title('Contour Plot of m_k and f')
        plt.legend()
        plt.show()

    # print("Optimal H:", H)


    return H
