"""
Manifold Samping
Given a user-provided blackbox function F that depends on an n-dimensional
vector x and returning m scalar values, this code solves the structured
blackbox optimization problem

min f(X)= h(F(x))

where F is smooth but expensive to evaluate and h is nonsmooth but cheap to evaluate
"""

from .build_p_models import build_p_models
from .call_user_scripts import call_user_scripts
from .check_inputs_and_initialize import check_inputs_and_initialize
from .choose_generator_set import choose_generator_set
from .evaluate_points_to_force_valid_model import evaluate_points_to_force_valid_model
from .load_tests import load_tests  # Used for automatic unittest-based discovery by main package
from .manifold_sampling_primal import manifold_sampling_primal
from .minimize_affine_envelope import minimize_affine_envelope
from .update_models import update_models
