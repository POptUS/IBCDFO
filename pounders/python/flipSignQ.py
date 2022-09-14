import numpy as np

def flipSignQ(Q, R, i, j):
    for k in range(i, j+1):
        idx = np.argmax(np.abs(Q[:, k]))
        if Q[idx, k] < 0:
            Q[:, k] = -Q[:, k]
            R[k, :] = -R[k, :]
    return [Q, R]