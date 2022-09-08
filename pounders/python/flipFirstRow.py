def flipFirstRow(Q, R, i, j):
    for k in range(i, j+1):
        if Q[0, k] < 0:
            Q[:, k] = -Q[:, k]
            R[k, :] = -R[k, :]
    return [Q, R]