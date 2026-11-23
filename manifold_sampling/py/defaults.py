"""
IMPORTANT: If any changes are made to default values here or to the set of
configuration arguments, please review and update manifold sampling inline
documentation to confirm that it remains consistent.
"""

# ----- SETS OF DICT CONFIG KEYS
# * EXPECTED_* implies that users have to provide this and only this set of keys
# * ALL_* implies that users can at most provide this set of keys
EXPECTED_PRIOR_KEYS = {"nfs", "X_init", "F_init", "xk_in"}
ALL_OPTIONS_KEYS = {
    "printf",
    "delta_0",
    "delta_max",
    "delta_min",
    "g_tol",
    "eta_1",
    "eta_3",
    "gamma_dec",
    "gamma_inc",
    "hfun_test_mode",
    "subprob_switch",
}


def compute_default_options():
    """
    This is intended for private, internal use only.  No error checking of
    returned values is performed by this function.

    :return: ``dict`` of full set of **Options** values set to default values
    """
    defaults = {
        "printf": 1,
        "delta_0": 0.1,
        "delta_max": 1e8,
        "delta_min": 1e-13,
        "g_tol": 1e-13,
        "eta_1": 0.01,
        "eta_3": 0.5,
        "gamma_dec": 0.5,
        "gamma_inc": 2,
        "hfun_test_mode": 1,
        "subprob_switch": "linprog",
    }
    assert set(defaults) == ALL_OPTIONS_KEYS
    return defaults
