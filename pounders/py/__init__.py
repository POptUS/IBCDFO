"""
Common and useful h functions for use with POUNDERS as well as functions for
combining models.
"""

# ----- PUBLIC INTERFACE
# fmt: off
from .general_h_funs import (
    identity_combine,
    leastsquares,
    neg_leastsquares,
    emittance_combine, emittance_h,
    squared_diff_from_mean
)

# General h function naming rules:
#  - h_* for h functions
#  - combine_* for associated combine model functions
# from .general_h_funs import (
#     combine_identity,
#     combine_leastsquares, h_leastsquares,
#     combine_neg_leastsquares,
#     combine_emittance, h_emittance,
#     combine_squared_diff_from_mean
# )

# -- Python unittest-based test framework
# Used for automatic test discovery by main package
from .load_tests import load_tests

# ----- INTERNAL INTERFACE
# This is used by tests only at present
from .phi2eval import phi2eval as _phi2eval

# Manifold sampling is using these directly by importing from the package rather
# than using relative imports, which makes some sense since that method is
# supposed to be developed independently of POUNDERS in its own isolated folder.
#
# Put in internal interface with the idea that any IBCDFO subpackage can access
# and use it.
from .checkinputss import checkinputss as _checkinputss
from .bmpts import bmpts as _bmpts
from .formquad import formquad as _formquad
