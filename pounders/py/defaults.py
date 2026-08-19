"""
IMPORTANT: If any changes are made to default values here or to the set of
configuration arguments, please review and update |pounders| inline
documentation to confirm that it remains consistent.
"""

import numpy as np

from .constants import TRSP_SOLVER_MINQ5
from .general_h_funs import h_leastsquares, combine_leastsquares

# ----- SETS OF DICT CONFIG KEYS
# * EXPECTED_* implies that users have to provide this and only this set of keys
# * ALL_* implies that users can at most provide this set of keys
EXPECTED_PRIOR_KEYS = {"nfs", "X_init", "F_init", "xk_in"}
ALL_MODEL_KEYS = {"np_max", "Par"}
ALL_OPTIONS_KEYS = {
    "printf",
    "spsolver",
    "delta_max",
    "delta_min",
    "delta_inact",
    "gamma_dec",
    "gamma_inc",
    "eta1",
    "hfun",
    "combinemodels",
}


def compute_default_prior(n, m):
    """
    This is intended for private, internal use only.  No error checking of
    inputs or returned values is performed by this function.

    These default values are unlikely to change in the future.

    :param n: Dimension (number of continuous, real-valued input variables)
    :param m: Dimension of output of **Ffun** (number of component functions)
    :return: ``dict`` of full set of **Prior** configuration values set to
        default values derived from given inputs
    """
    defaults = {
        "nfs": 0,
        "X_init": np.full((0, n), np.nan, float),
        "F_init": np.full((0, m), np.nan, float),
        "xk_in": 0,
    }
    assert set(defaults) == EXPECTED_PRIOR_KEYS
    return defaults


def compute_default_model(n):
    """
    This is intended for private, internal use only.  No error checking of
    inputs or returned values is performed by this function.

    These default values are unlikely to change in the future.

    :param n: Dimension (number of continuous, real-valued input variables)
    :return: ``dict`` of full set of **Model** configuration values set to
        default values derived from given input
    """
    defaults = {
        "np_max": 2 * n + 1,
        "Par": [np.sqrt(n), np.maximum(10, np.sqrt(n)), 1.0e-3, 1.0e-3, 0],
    }
    assert set(defaults) == ALL_MODEL_KEYS
    return defaults


def compute_default_options(delta_0, g_tol, Low, Upp):
    r"""
    This is intended for private, internal use only.  No error checking of
    inputs or returned values is performed by this function.

    Except for the default **spsolver** these default values are unlikely to
    change in the future.

    :param delta_0: Positive initial trust region radius
    :param g_tol: Tolerance for the 2-norm of the model gradient
    :param Low: :math:`\np`-element 1D NumPy array of lower bounds
    :param Upp: :math:`\np`-element 1D NumPy array of upper bounds
    :return: ``dict`` of full set of ``Options`` values set to default values
        derived from given inputs
    """
    defaults = {
        "printf": 0,
        "spsolver": TRSP_SOLVER_MINQ5,
        "delta_max": np.minimum(0.5 * np.min(Upp - Low), 1.0e3 * delta_0),
        "delta_min": np.minimum(delta_0 * 1.0e-13, 0.1 * g_tol),
        "delta_inact": 0.75,
        "gamma_dec": 0.5,
        "gamma_inc": 2,
        "eta1": 0.05,
        "hfun": h_leastsquares,
        "combinemodels": combine_leastsquares,
    }
    assert set(defaults) == ALL_OPTIONS_KEYS
    return defaults
