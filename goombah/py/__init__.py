"""
GOOMBAH: a trust-region-based method for composite nonsmooth optimization
problems.
"""

from .minimize_quadratics_lhs import minimize_quadratics_lhs

# -- Python unittest-based test framework
# Used for automatic test discovery by main package
from .load_tests import load_tests
