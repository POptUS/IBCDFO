"""
IBCDFO: Interpolation-Based Composite Derivative-Free Optimization
This package contains methods to solve structured blackbox optimization
problems of the form: 
    minimize h(F(x)) 
where x is the n-dimensional optimization variable, F(x) is the m-dimensional
output of blackbox, and h is a known scalar-valued mapping.
"""

from importlib.metadata import version

# Follow typical version-access interface used by other packages
# (e.g., numpy, scipy, pandas, matplotlib)
__version__ = version("ibcdfo")

from . import pounders
