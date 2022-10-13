import numpy as np

# import boxline function from boxline.py
from boxline import boxline


def bmpts(X, Modeld, Low, Upp, delta, theta):
    [num, n] = np.shape(Modeld)
    # For each ray, find largest t to remain feasible
    T = np.zeros((2, num))
    for j in range(0, num):
        T[0, j] = boxline(delta * Modeld[j], X, Low, Upp)
        T[1, j] = boxline(-delta * Modeld[j], X, Low, Upp)
    # Safe to use our directions
    Modeld = Modeld.astype('float64')  # Change uint8 to float
    if np.min(np.max(T, axis=0)) >= theta:
        mp = n - num
        for j in range(0, num):
            if T[0, j] >= T[1, j]:
                Modeld[j] = delta * Modeld[j] * T[0, j]
            else:
                Modeld[j] = -delta * Modeld[j] * T[1, j]
    else:
        # May want to turn this display off
        print('Note: Geometry points need to be coordinate directions!')
        mp = 0
        Modeld = np.zeros((n, n))
        for j in range(0, n):
            t1 = min(X[j] - Low[j], delta)
            t2 = min(Upp[j] - X[j], delta)
            if t1 >= t2:
                Modeld[j, j] = -t1
            else:
                Modeld[j, j] = t2
    return Modeld, mp
