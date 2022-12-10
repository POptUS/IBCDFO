import numpy as np


def neg_leastsquares(Cres, Gres, Hres):
    G, H = leastsquares(Cres, Gres, Hres)
    G = -G
    H = -H

    return G, H


def leastsquares(Cres, Gres, Hres):
    n, _, m = Hres.shape

    G = 2 * Gres @ Cres.T
    H = np.zeros((n, n))
    for i in range(m):
        H = H + Cres[i] * Hres[:, :, i]

    H = 2 * H + 2 * Gres @ Gres.T

    return G, H


def emittance_combine(Cres, Gres, Hres):

    n, _, m = Hres.shape

    assert m == 3, "Emittance calculation requires exactly three quantities"

    G = Cres[0] * Gres[:, 1] + Cres[1] * Gres[:, 0] - 2 * Cres[2] * Gres[:, 2]
    H = Cres[0] * Hres[:, :, 1] + Cres[1] * Hres[:, :, 0] + np.outer(Gres[:, 1], Gres[:, 0]) + np.outer(Gres[:, 0], Gres[:, 1]) - 2 * Cres[2] * Hres[:, :, 2] - 2 * np.outer(Gres[:, 2], Gres[:, 2])

    return G, H


def emittance_h(F):

    assert len(F) == 3, "Emittance must have exactly 3 inputs"
    h = F[0] * F[1] - F[2] ** 2

    return h


def squared_diff_from_mean(Cres, Gres, Hres):
    # Combines models for the following h function
    #    h = @(F)sum((F - 1/m*sum(F)).^2) - alpha*(1/m*sum(F))^2
    # That is, the objective is to have the vector close to it's mean, and have
    # a small mean (penalized by alpha)
    alpha = 0

    n, _, m = Hres.shape

    m_sumF = np.mean(Cres)
    m_sumG = 1 / m * np.sum(Gres, axis=1)
    sumH = np.sum(Hres, axis=2)

    G = np.zeros(n)
    for i in range(m):
        G = G + (Cres[i] - m_sumF) * (Gres[:, i] - m_sumG)
    G = 2 * G - 2 * alpha * m_sumF * m_sumG

    H = np.zeros((n, n))
    for i in range(m):
        H = H + (Cres[i] - m_sumF) * (Hres[:, :, i] + sumH) + np.outer(Gres[:, i] - m_sumG, Gres[:, i] - m_sumG)

    H = 2 * H

    H = H - (2 * alpha / m) * m_sumF * sumH - (2 * alpha) * np.outer(m_sumG, m_sumG)

    return G, H
