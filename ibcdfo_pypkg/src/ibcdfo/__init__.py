"""
IBCDFO: Interpolation-Based Composite Derivative-Free Optimization
This package contains methods to solve structured blackbox optimization
problems of the form:
    minimize h(F(x))
where x is the n-dimensional optimization variable, F(x) is the m-dimensional
output of blackbox, and h is a known scalar-valued mapping.
"""

from importlib.metadata import version

__version__ = version("ibcdfo")

from . import pounders
from . import manifold_sampling

# ----- Python unittest-based test framework
# Used for automatic test discovery
from .load_tests import load_tests

# Allow users to run full test suite as mytemplate.test()
from .test import test
