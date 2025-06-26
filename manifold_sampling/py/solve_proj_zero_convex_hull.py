import cvxpy as cp
import numpy as np

def solve_proj_zero_convex_hull(G):

    # Example matrix G (replace with your data)
    # G has shape (n, m): n-dimensional column vectors, m vectors total
    # G = np.array([
    #     [1.0, 0.0, 0.5],
    #     [0.0, 1.0, 0.5]
    # ])  # Shape (2, 3)

    
    n, m = G.shape

    # Optimization variable: convex coefficients
    lmbda = cp.Variable(m)

    # Convex combination of the columns
    point = G @ lmbda

    # Objective: minimize the squared distance to origin
    objective = cp.Minimize(cp.sum_squares(point))

    # Constraints: convex combination
    constraints = [
        lmbda >= 0,
        cp.sum(lmbda) == 1
    ]

    # Problem definition and solve
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Results
    print("Optimal lambda:", lmbda.value)
    print("Projected point:", G @ lmbda.value)
    print("Distance to origin:", np.linalg.norm(G @ lmbda.value))
    return lmbda.value