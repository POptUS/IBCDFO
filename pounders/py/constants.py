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
