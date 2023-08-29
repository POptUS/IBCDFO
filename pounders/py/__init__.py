"""
POUNDERS: Practical Optimization Using No Derivatives for sums of Squares
Given a user-provided blackbox function F that depends on an n-dimensional
vector x and returning m scalar values, this code solves the structured
blackbox optimization problem
min { f(X)=sum_(i=1:m) F_i(x)^2, such that L_j <= x_j <= U_j, j=1,...,n }.
"""

from .bmpts import bmpts
from .boxline import boxline
from .bqmin import bqmin
from .checkinputss import checkinputss
from .flipFirstRow import flipFirstRow
from .flipSignQ import flipSignQ
from .formquad import formquad
from .general_h_funs import (
    emittance_combine,
    emittance_h,
    identity_combine,
    leastsquares,
    neg_leastsquares,
    squared_diff_from_mean,
)
from .phi2eval import phi2eval
from .pounders import pounders
from .prepare_outputs_before_return import prepare_outputs_before_return

# ----- Python unittest-based test framework
# Used for automatic test discovery by main package
from .load_tests import load_tests
