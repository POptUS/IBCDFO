import warnings

import numpy as np

# Below this a snap is not worth mentioning: it is the slop that a solver
# leaves on a step it considers feasible. Deliberately not exposed, since no
# caller should have to reason about it.
_NOISE_TOL = 1.0e3 * np.finfo(float).eps


def snap_to_bounds(x, Low, Upp, snap_tol, what):
    """
    Put ``x`` back inside ``[Low, Upp]``, complaining if it was not already
    there.

    Everything checked here comes from a subproblem solver or a user script,
    so a large violation means something is wrong and silently snapping would
    hide it (it has hidden real bugs before). How far outside decides what
    happens:

    * up to ~2e-13: snap, and say nothing
    * up to **snap_tol**: warn, then snap
    * beyond that: raise :class:`ValueError`

    The first threshold is fixed, so setting **snap_tol** below it buys
    nothing: violations that small are always snapped in silence.

    This compares absolute violations and does not scale with ``x``. On a
    problem whose variables are large, one unit in the last place of
    ``x`` already exceeds the default **snap_tol**, so a correct solver will
    trip this check on rounding alone unless **snap_tol** is raised to suit.

    :param x: NumPy array holding the point or step to check; a row per point.
    :param Low: NumPy array of lower bounds on ``x``
    :param Upp: NumPy array of upper bounds on ``x``
    :param snap_tol: Largest violation that is snapped rather than refused.
        The solvers default this to 1e-6, the feasibility tolerance that
        LP/QP solvers such as MATLAB's ``linprog`` promise; tighter values
        reject their output.
    :param what: ``str`` describing ``x``, for the error/warning message
    :return: ``x`` snapped into ``[Low, Upp]``
    """
    assert isinstance(x, np.ndarray)
    assert isinstance(Low, np.ndarray) and isinstance(Upp, np.ndarray)
    assert snap_tol >= 0.0

    viol = max(np.max(Low - x), np.max(x - Upp))

    if viol > _NOISE_TOL:
        msg = f"{what} violates its bounds by {viol:g}"
        if viol > snap_tol:
            raise ValueError(f"{msg}, which exceeds snap_tol = {snap_tol:g}.")
        warnings.warn(f"{msg}; snapping it onto the bounds.", stacklevel=2)

    return np.minimum(Upp, np.maximum(Low, x))
