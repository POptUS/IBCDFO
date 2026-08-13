import numpy as np

from .general_h_funs import h_leastsquares, combine_leastsquares


def compute_default_options(delta_0, g_tol, Low, Upp):
    return {
        "printf": 0,
        "spsolver": 2,
        "delta_max": np.minimum(0.5 * np.min(Upp - Low), 1.0e3 * delta_0),
        "delta_min": np.minimum(delta_0 * 1.0e-13, 0.1 * g_tol),
        "delta_inact": 0.75,
        "gamma_dec": 0.5,
        "gamma_inc": 2,
        "eta1": 0.05,
        "hfun": h_leastsquares,
        "combinemodels": combine_leastsquares,
    }
