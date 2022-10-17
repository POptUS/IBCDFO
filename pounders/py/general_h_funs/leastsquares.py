import numpy as np

def leastsquares(Cres,Gres,Hres):
    n, _, m = Hres.shape

    G = 2 * Gres @ Cres.T
    H = np.zeros((n,n))
    for i in range(m):
        H = H + Cres[i]*Hres[:,:,i]

    H = 2*H + 2*Gres @ Gres.T

    return G, H
