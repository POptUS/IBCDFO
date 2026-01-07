"""
Common and useful h functions for use with the Manifold Sampling method
"""

# ----- PUBLIC INTERFACE
# fmt: off
from .general_nonsmooth_h_funs import (
    h_censored_L1_loss,
    h_max_gamma_over_KY,
    h_one_norm,
    h_pw_minimum,
    h_pw_minimum_squared,
    h_pw_maximum,
    h_pw_maximum_squared,
    h_piecewise_quadratic,
    h_quantile,
    max_plus_quadratic_violation_penalty
)
# fmt: on

# -- Python unittest-based test framework
# Used for automatic test discovery by main package
from .load_tests import load_tests
