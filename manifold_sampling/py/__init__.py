"""
Common and useful h functions for use with the Manifold Sampling method
"""

# ----- PUBLIC INTERFACE
# fmt: off
from .h_examples import (
    censored_L1_loss,
    max_gamma_over_KY,
    one_norm,
    pw_minimum, pw_minimum_squared,
    pw_maximum, pw_maximum_squared,
    piecewise_quadratic,
    quantile
)

# -- Python unittest-based test framework
# Used for automatic test discovery by main package
from .load_tests import load_tests
