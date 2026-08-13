import numpy as np


def compute_default_model(n):
    return {"np_max": 2 * n + 1, "Par": [np.sqrt(n), np.maximum(10, np.sqrt(n)), 10**-3, 0.001, 0]}
