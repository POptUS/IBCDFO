# ----- TRUST-REGION SUBPROBLEM SOLVERS
# Ensure that these match the analogous constants implemented for
# POUNDERS/MATLAB.
#
# Both MATLAB and Python implementations should declare the union of all solvers
# available even if they don't support one or more of the solvers.
TRSP_SOLVER_SIMPLE = 1
TRSP_SOLVER_MINQ5 = 2
TRSP_SOLVER_MINQ8 = 3

# ----- ERROR & WARNING MESSAGES
WARNING_SIMPLE_TRSP = "The simple TRSP solver should only be used for testing or debugging"

# ----- SETS OF DICT CONFIG KEYS
# * ALL_* implies that users can at most provide this set of keys
# * EXPECTED_* implies that users have to provide this and only this set of keys
ALL_MODEL_KEYS = {"np_max", "Par"}
EXPECTED_PRIOR_KEYS = {"nfs", "X_init", "F_init", "xk_in"}
ALL_OPTIONS_KEYS = {"printf", "spsolver", "delta_max", "delta_min", "delta_inact", "gamma_dec", "gamma_inc", "eta1", "hfun", "combinemodels"}
