import sys
import os
import time
import contextlib
import io
import inspect
import numpy as np

try:
    import ipdb
except ModuleNotFoundError:
    ipdb = None

try:
    from .prepare_outputs_before_return_gradient import prepare_outputs_before_return
except ImportError:
    from prepare_outputs_before_return_gradient import prepare_outputs_before_return

try:
    import poptus
except ModuleNotFoundError:
    class _NoOpArchiver:
        results_group = None

    class _PoptusFallback:
        LOG_LEVEL_DEFAULT = 0
        LOG_LEVEL_MIN_DEBUG = 0
        Hdf5Archiver = _NoOpArchiver

    poptus = _PoptusFallback()


PYROL_INNER_ITERATION_LIMIT = 5
PYROL_INNER_GRADIENT_TOLERANCE = 1e-6
PYROL_INNER_STEP_TOLERANCE = 1e-12

# Populated by the most recent pouders(...) run.  Keeping this as module-level
# state avoids changing POUNDERS' return signature while making notebook plots
# reproducible.
last_progress_history = []
last_fpr_history = []
last_run_metadata = {}


def _max_step_radius_from_bounds(Lows, Upps):
    finite_bounds = np.concatenate(
        (
            np.abs(np.asarray(Lows, dtype=float).reshape(-1)),
            np.abs(np.asarray(Upps, dtype=float).reshape(-1)),
        )
    )
    finite_bounds = finite_bounds[np.isfinite(finite_bounds)]
    if finite_bounds.size == 0:
        return 1.0

    radius = float(np.max(finite_bounds))
    if radius <= 0:
        return 1.0
    return radius


def _solve_trsp_pyrol(G, H, Lows, Upps, n):
    """Solve the bound-constrained trust-region subproblem using PyROL."""
    max_inner_radius = _max_step_radius_from_bounds(Lows, Upps)
    try:
        
        from pyrol import Objective, ParameterList, Problem, Bounds, Solver, getCout
        from pyrol.vectors import NumPyVector as npVector
        print("----------------------pyrol imported------------------")
        class TRSPObjective(Objective):
            def __init__(self, G, H):
                super().__init__()
                self.G = np.asarray(G, dtype=float).reshape(-1)
                self.H = np.asarray(H, dtype=float)

            def value(self, x, tol):
                s = np.asarray(x[:], dtype=float)
                return self.G @ s + 0.5 * s @ (self.H @ s)

            def gradient(self, g, x, tol):
                s = np.asarray(x[:], dtype=float)
                g[:] = self.G + self.H @ s

            def hessVec(self, hv, v, x, tol):
                hv[:] = self.H @ np.asarray(v[:], dtype=float)

        x = npVector(np.zeros(n))
        g = x.dual()

        objective = TRSPObjective(G, H)
        problem = Problem(objective, x, g)

        lower = npVector(np.asarray(Lows, dtype=float).reshape(-1))
        upper = npVector(np.asarray(Upps, dtype=float).reshape(-1))
        problem.addBoundConstraint(Bounds(lower, upper))

        p = ParameterList()
        p["General"] = ParameterList()
        p["General"]["Output Level"] = 0
        p["Step"] = ParameterList()
        p["Step"]["Trust Region"] = ParameterList()
        p["Step"]["Trust Region"]["Subproblem Solver"] = "Truncated CG"
        p["Step"]["Trust Region"]["Subproblem Model"] = "Lin-More"
        p["Step"]["Trust Region"]["Initial Radius"] = max_inner_radius
        p["Step"]["Trust Region"]["Maximum Radius"] = 1
        p["Status Test"] = ParameterList()
        p["Status Test"]["Iteration Limit"] = PYROL_INNER_ITERATION_LIMIT
        p["Status Test"]["Gradient Tolerance"] = PYROL_INNER_GRADIENT_TOLERANCE
        p["Status Test"]["Step Tolerance"] = PYROL_INNER_STEP_TOLERANCE

        solver = Solver(problem, p)
        solver.solve(getCout())

        Xsp = np.asarray(x[:], dtype=float).reshape(n, 1)
        mdec = objective.value(x, 0.0)

        return Xsp, mdec
    except ModuleNotFoundError:
        pass

    try:
        print("----------------------rol imported------------------")
        import ROL
        from ROL.numpy_vector import NumpyVector
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "spsolver=3 requires PyROL/ROL. Install pyroltrilinos in this "
            "Python environment."
        ) from exc

    def numpy_vector(values):
        values = np.asarray(values, dtype=float).reshape(-1)
        vec = NumpyVector(values.size)
        vec.data[:] = values
        return vec

    class TRSPObjective(ROL.Objective):
        def __init__(self, G, H):
            ROL.Objective.__init__(self)
            self.G = np.asarray(G, dtype=float).reshape(-1)
            self.H = np.asarray(H, dtype=float)

        def value(self, x, tol):
            s = np.asarray(x.data, dtype=float)
            return float(self.G @ s + 0.5 * s @ (self.H @ s))

        def gradient(self, g, x, tol):
            s = np.asarray(x.data, dtype=float)
            g.data[:] = self.G + self.H @ s

        def hessVec(self, hv, v, x, tol):
            hv.data[:] = self.H @ np.asarray(v.data, dtype=float)

    objective = TRSPObjective(G, H)
    x = numpy_vector(np.zeros(n))
    lower = numpy_vector(Lows)
    upper = numpy_vector(Upps)

    params = ROL.ParameterList(
        {
            "Step": {
                "Type": "Trust Region",
                "Trust Region": {
                    "Subproblem Solver": "Truncated CG",
                    "Initial Radius": max_inner_radius,
                    "Maximum Radius": 1,
                },
            },
            "General": {"Print Verbosity": 0},
            "Status Test": {
                "Iteration Limit": PYROL_INNER_ITERATION_LIMIT,
                "Gradient Tolerance": PYROL_INNER_GRADIENT_TOLERANCE,
                "Step Tolerance": PYROL_INNER_STEP_TOLERANCE,
            },
        },
        "Parameters",
    )
    problem = ROL.OptimizationProblem(objective, x, ROL.Bounds(lower, upper))
    solver = ROL.OptimizationSolver(problem, params)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve()

    Xsp = np.asarray(x.data, dtype=float).reshape(n, 1)
    mdec = objective.value(x, 0.0)

    return Xsp, mdec


def pouders(
    fun,
    X0,
    n,
    nfmax,
    gtol,
    delta,
    m,
    L,
    U,
    logger,
    spsolver=3,
    hfun=None,
    combinemodels=None,
    fpr_reduction=None,
    residuals_per_circuit=1,
    shots_per_circuit=1,
    fpr_use_union_mask=False,
    rho_uses_full_objective=False,
    iter_callback=None,
):
    """
    POUDERS: Practical Optimization Using Derivatives for sums of Squares
      [X,F,flag,xkin] = ...
           pouders(fun,X0,n,npmax,nfmax,gtol,delta,nfs,m,F0,xkin,L,U,printf)

    This code minimizes output from a structured blackbox function, solving
    min { f(X)=sum_(i=1:m) F_i(x)^2, such that L_j <= X_j <= U_j, j=1,...,n }
    where the user-provided blackbox F is specified in the handle fun. Evaluation
    of this F must result in the return of a 1-by-m row vector. Bounds must be
    specified in U and L but can be set to L=-Inf(1,n) and U=Inf(1,n) if the
    unconstrained solution is desired. The algorithm will not evaluate F
    outside of these bounds, but it is possible to take advantage of function
    values at infeasible X if these are passed initially through (X0,F0).
    In each iteration, the algorithm forms an interpolating quadratic model
    of the function and minimizes it in an infinity-norm trust region.

    This software comes with no warranty, is not bug-free, and is not for
    industrial use or public distribution.
    Direct requests and bugs to wild@mcs.anl.gov.
    A technical report/manual is forthcoming, a brief description is in
    Nuclear Energy Density Optimization. Phys. Rev. C, 82:024313, 2010.

    --INPUTS-----------------------------------------------------------------
    fun     [f h] Function handle so that fun(x) evaluates F (@calfun)
    X0      [dbl] [max(nfs,1)-by-n] Set of initial points  (zeros(1,n))
    n       [int] Dimension (number of continuous variables)
    nfmax   [int] Maximum number of function evaluations (>n+1) (100)
    gtol    [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
    delta   [dbl] Positive trust region radius (.1)
    m       [int] Number of residual components
    L       [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
    U       [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
    logger  [obj] POptUS logger object 
    spsolver [int] Trust-region subproblem solver flag. Use 3 for PyROL.

    Optionally, a user can specify and outer-function that maps the the elements
    of F to a scalar value (to be minimized). Doing this also requires a function
    handle (combinemodels) that tells pounders how to map the linear and
    quadratic terms from the residual models into a single quadratic TRSP model.

    hfun           [f h] Function handle for mapping output from F
    combinemodels  [f h] Function handle for combine residual models
    --OUTPUTS----------------------------------------------------------------
    X       [dbl] [nfmax+nfs-by-n] Locations of evaluated points
    F       [dbl] [nfmax+nfs-by-m] Function values of evaluated points
    flag    [dbl] Termination criteria flag:
                  = 0 normal termination because of grad,
                  > 0 exceeded nfmax evals,   flag = norm of grad at final X
                  = -1 if input was fatally incorrect (error message shown)
                  = -2 if a valid model produced X[nf] == X[xkin] or (mdec == 0, Fs[nf] == Fs[xkin])
                  = -3 error if a NaN was encountered
                  = -4 error in TRSP Solver
                  = -5 unable to get model improvement with current parameters
                  = -6 adaptive measurement-shot budget exhausted
    xkin    [int] Index of point in X representing approximate minimizer
    """

    archiver = poptus.Hdf5Archiver()
    group = archiver.results_group

    def log(msg):
        logger.log("POUDERS", msg, poptus.LOG_LEVEL_DEFAULT)

    def log_debug(msg, level):
        logger.log("POUDERS", msg, poptus.LOG_LEVEL_MIN_DEBUG + level)

    global last_progress_history, last_fpr_history, last_run_metadata
    residuals_per_circuit = int(residuals_per_circuit)
    if residuals_per_circuit <= 0:
        raise ValueError("residuals_per_circuit must be positive.")
    _shots_is_callable = callable(shots_per_circuit)
    if not _shots_is_callable:
        shots_per_circuit = float(shots_per_circuit)
        if not np.isfinite(shots_per_circuit) or shots_per_circuit < 0:
            raise ValueError("shots_per_circuit must be a nonnegative finite number or a callable.")
    _total_circuits = int(np.ceil(m / residuals_per_circuit))

    def _shots_for(circuit_count):
        # Total shots for `circuit_count` circuits.  Scalar -> circuit_count * scalar
        # (uniform runs).  Callable (adaptive runs) returns the CURRENT actual total
        # shots, so the per-circuit count is ignored and the true total is recorded.
        if _shots_is_callable:
            return float(shots_per_circuit())
        return float(circuit_count) * float(shots_per_circuit)

    progress_history = []
    last_progress_history = progress_history
    last_run_metadata = {
        "m": int(m),
        "n": int(n),
        "residuals_per_circuit": residuals_per_circuit,
        "total_circuits": int(np.ceil(m / residuals_per_circuit)),
        "shots_per_circuit": (None if _shots_is_callable else shots_per_circuit),
        "fpr_enabled": fpr_reduction is not None,
        "fpr_use_union_mask": bool(fpr_use_union_mask),
        "rho_uses_full_objective": bool(rho_uses_full_objective),
    }

    if hfun is None:

        def hfun(F):
            return np.sum(F**2)

    if combinemodels is None:
        try:
            from .general_h_funs import combine_leastsquares as combinemodels
        except ImportError:
            from general_h_funs import combine_leastsquares as combinemodels

    def _callable_accepts_n_positional_args(callable_obj, nargs):
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False
        parameters = list(signature.parameters.values())
        for parameter in parameters:
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                return True
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return len(positional) >= nargs

    def _call_hfun(Fval, info=None):
        if _callable_accepts_n_positional_args(hfun, 2):
            try:
                signature = inspect.signature(hfun)
                parameters = list(signature.parameters.values())
                positional = [
                    parameter
                    for parameter in parameters
                    if parameter.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
                second_name = positional[1].name if len(positional) >= 2 else None
            except (TypeError, ValueError):
                second_name = None

            if second_name == "counts" and isinstance(info, dict) and "counts" in info:
                return hfun(Fval, info["counts"])
            return hfun(Fval, info)
        return hfun(Fval)

    def _call_combinemodels(Cres, Gres, Hres, info=None):
        if _callable_accepts_n_positional_args(combinemodels, 4):
            return combinemodels(Cres, Gres, Hres, info)
        return combinemodels(Cres, Gres, Hres)

    hfun_uses_info = _callable_accepts_n_positional_args(hfun, 2)
    combinemodels_uses_info = _callable_accepts_n_positional_args(combinemodels, 4)

    def _fun_accepts_keyword(keyword):
        try:
            signature = inspect.signature(fun)
        except (TypeError, ValueError):
            return True
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return keyword in signature.parameters

    fun_accepts_info = _fun_accepts_keyword("return_info")

    def _evaluate_fun(x):
        kwargs = {}
        if (hfun_uses_info or combinemodels_uses_info) and fun_accepts_info:
            kwargs["return_info"] = True

        output = fun(x, **kwargs) if kwargs else fun(x)
        if not isinstance(output, tuple):
            raise TypeError("The objective function must return (F, J) or (F, J, info).")
        if len(output) == 3:
            Fval, Jval, info = output
        elif len(output) == 2:
            Fval, Jval = output
            info = None
        else:
            raise ValueError("The objective function must return (F, J) or (F, J, info).")
        if isinstance(info, dict):
            # The Jacobian is returned separately as Jval.  Keeping duplicate
            # Jacobian arrays in info can double memory use for GST problems.
            info = dict(info)
            info.pop("jacobian", None)
            info.pop("jacobian_residuals_by_parameters", None)
        return Fval, Jval, info

    fpr_history = []
    last_fpr_history = fpr_history
    fpr_union_mask = np.zeros(m, dtype=bool) if fpr_reduction is not None else None
    revealed_residual_mask = np.zeros(m, dtype=bool)
    cumulative_eval_residuals = 0

    def _normalize_fpr_mask(mask):
        if fpr_reduction is None:
            return None
        if isinstance(mask, tuple):
            mask = mask[0]
        arr = np.asarray(mask)
        if arr.dtype == bool:
            arr = arr.reshape(-1)
            if arr.size != m:
                raise ValueError(
                    f"FPR reduction returned a boolean mask of length {arr.size}, expected {m}."
                )
            keep = arr.copy()
        else:
            indices = np.asarray(mask, dtype=int).reshape(-1)
            if np.any(indices < 0) or np.any(indices >= m):
                raise ValueError("FPR reduction returned residual indices outside [0, m).")
            keep = np.zeros(m, dtype=bool)
            keep[indices] = True
        if not np.any(keep):
            raise ValueError("FPR reduction selected zero residual entries.")
        return keep

    def _fpr_mask_for_x(x):
        if fpr_reduction is None:
            return None
        keep = _normalize_fpr_mask(fpr_reduction(np.asarray(x, dtype=float).copy()))
        previous_union = fpr_union_mask.copy()
        new_keep = np.logical_and(keep, np.logical_not(previous_union))
        fpr_union_mask[:] = np.logical_or(fpr_union_mask, keep)
        active_keep = fpr_union_mask.copy() if fpr_use_union_mask else keep.copy()
        entry = {
            "call_index": len(fpr_history),
            "selected_residuals": int(np.sum(keep)),
            "total_residuals": int(m),
            "new_residuals": int(np.sum(new_keep)),
            "union_residuals": int(np.sum(fpr_union_mask)),
            "active_residuals": int(np.sum(active_keep)),
            "selected_circuits": int(np.ceil(np.sum(keep) / residuals_per_circuit)),
            "new_circuits": int(np.ceil(np.sum(new_keep) / residuals_per_circuit)),
            "union_circuits": int(np.ceil(np.sum(fpr_union_mask) / residuals_per_circuit)),
            "active_circuits": int(np.ceil(np.sum(active_keep) / residuals_per_circuit)),
            "cumulative_shots_revealed": _shots_for(
                np.ceil(np.sum(fpr_union_mask) / residuals_per_circuit)
            ),
        }
        fpr_history.append(entry)
        if hasattr(fpr_reduction, "pounders_history"):
            fpr_reduction.pounders_history.append(entry)
        else:
            fpr_reduction.pounders_history = [entry]
        log(
            "FPR reduction selected "
            f"{entry['selected_residuals']}/{m} residuals; "
            f"union={entry['union_residuals']}/{m}; "
            f"active={entry['active_residuals']}/{m}."
        )
        return active_keep

    def _mask_info(info, keep):
        if info is None or keep is None or not isinstance(info, dict):
            return info
        masked = dict(info)
        for key, value in info.items():
            try:
                arr = np.asarray(value)
            except Exception:
                continue
            if arr.shape == (m,):
                arr = arr.copy()
                arr[~keep] = 0
                masked[key] = arr
        masked["fpr_mask"] = keep.copy()
        masked["fpr_selected_residuals"] = int(np.sum(keep))
        return masked

    def _apply_fpr_reduction(Fval, Jval, info, keep):
        Fflat = np.asarray(Fval, dtype=float).reshape(-1).copy()
        Jarr = np.asarray(Jval, dtype=float).copy()
        if Fflat.size != m or Jarr.shape != (n, m):
            raise ValueError(
                f"Objective returned F={Fflat.shape}, J={Jarr.shape}; expected F=({m},), J=({n}, {m})."
            )
        if keep is not None:
            Fflat[~keep] = 0.0
            Jarr[:, ~keep] = 0.0
            info = _mask_info(info, keep)
        return Fflat, Jarr, info

    def _count_to_circuits(count):
        return int(np.ceil(float(count) / residuals_per_circuit))

    def _record_progress(
        *,
        phase,
        nf_value,
        x_index,
        active_mask,
        active_objective,
        full_objective,
        incumbent_index,
        incumbent_active_objective,
        incumbent_full_objective,
        delta_value=None,
        ng_value=None,
        rho_value=None,
        rho_active_value=None,
        rho_full_value=None,
        mdec_value=None,
        step_norm_value=None,
        accepted=None,
    ):
        nonlocal cumulative_eval_residuals, revealed_residual_mask

        if active_mask is None:
            active_mask_arr = np.ones(m, dtype=bool)
        else:
            active_mask_arr = np.asarray(active_mask, dtype=bool).reshape(-1)
        active_residuals = int(np.sum(active_mask_arr))
        active_circuits = _count_to_circuits(active_residuals)

        newly_revealed_mask = np.logical_and(active_mask_arr, np.logical_not(revealed_residual_mask))
        newly_revealed_residuals = int(np.sum(newly_revealed_mask))
        revealed_residual_mask[:] = np.logical_or(revealed_residual_mask, active_mask_arr)
        union_revealed_residuals = int(np.sum(revealed_residual_mask))
        newly_revealed_circuits = _count_to_circuits(newly_revealed_residuals)
        union_revealed_circuits = _count_to_circuits(union_revealed_residuals)

        cumulative_eval_residuals += active_residuals
        latest_fpr = fpr_history[-1] if (fpr_reduction is not None and fpr_history) else {}

        entry = {
            "phase": phase,
            "nf": int(nf_value),
            "x_index": int(x_index),
            "incumbent_index": int(incumbent_index),
            "active_objective": float(active_objective),
            "full_objective": float(full_objective),
            "incumbent_active_objective": float(incumbent_active_objective),
            "incumbent_full_objective": float(incumbent_full_objective),
            "active_residuals": active_residuals,
            "active_circuits": active_circuits,
            "new_residuals_revealed": newly_revealed_residuals,
            "new_circuits_revealed": newly_revealed_circuits,
            "union_residuals_revealed": union_revealed_residuals,
            "union_circuits_revealed": union_revealed_circuits,
            "cumulative_eval_residuals": int(cumulative_eval_residuals),
            "cumulative_eval_circuits": _count_to_circuits(cumulative_eval_residuals),
            "cumulative_shots_revealed": _shots_for(union_revealed_circuits),
            "cumulative_eval_shots": _shots_for(_count_to_circuits(cumulative_eval_residuals)),
            "shots_per_circuit": (_shots_for(1) / max(_total_circuits, 1)
                                  if _shots_is_callable else float(shots_per_circuit)),
            "fpr_enabled": fpr_reduction is not None,
            "fpr_use_union_mask": bool(fpr_use_union_mask),
            "fpr_selected_residuals": latest_fpr.get("selected_residuals", active_residuals),
            "fpr_selected_circuits": latest_fpr.get("selected_circuits", active_circuits),
            "fpr_new_residuals": latest_fpr.get("new_residuals", newly_revealed_residuals),
            "fpr_new_circuits": latest_fpr.get("new_circuits", newly_revealed_circuits),
            "fpr_union_residuals": latest_fpr.get("union_residuals", union_revealed_residuals),
            "fpr_union_circuits": latest_fpr.get("union_circuits", union_revealed_circuits),
            "delta": None if delta_value is None else float(delta_value),
            "ng": None if ng_value is None else float(ng_value),
            "rho": None if rho_value is None else float(rho_value),
            "rho_active": None if rho_active_value is None else float(rho_active_value),
            "rho_full": None if rho_full_value is None else float(rho_full_value),
            "rho_uses_full_objective": bool(rho_uses_full_objective),
            "mdec": None if mdec_value is None else float(mdec_value),
            "step_norm": None if step_norm_value is None else float(step_norm_value),
            "accepted": accepted,
        }
        progress_history.append(entry)
        return entry

    # The trust-region subproblem is solved with PyROL/ROL only.
    if spsolver != 3:
        raise ValueError("Only spsolver=3 (PyROL/ROL) is supported.")

    maxdelta = min(0.5 * np.min(U - L), (10**3) * delta)
    mindelta = min(delta * (10**-13), gtol / 10)
    gam0 = 0.5
    gam1 = 2
    eta1 = 0.05

    eps = np.finfo(float).eps  # Define machine epsilon
    log("Beginning gradient-based optimization.")
    X = np.vstack((X0, np.zeros((nfmax - 1, n))))
    F = np.zeros((nfmax, m))
    J = [None] * nfmax
    # Keep full unmasked data only for the active center.  Storing every raw
    # GST Jacobian would require roughly hundreds of MB per evaluation.
    center_raw_F = None
    center_raw_J = None
    center_raw_Info = None
    Info = [None] * nfmax
    Fs = np.zeros(nfmax)      # Objective value used by the local POUNDERS model.
    FullFs = np.zeros(nfmax)  # Full objective value for reporting/comparison.
    nf = 0  # in Matlab this is 1
    xkin = 0

    # first evaluation:
    t0 = time.perf_counter()
    F0, J0, eval_info = _evaluate_fun(X[nf])
    log(f"Initial residual/Jacobian evaluation took {time.perf_counter() - t0:.2f} seconds.")
    F0 = np.asarray(F0, dtype=float).reshape(-1)
    J0 = np.asarray(J0, dtype=float)

    if F0.size != m:
        X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, -1)
        # TODO: Should this use logger.warn or logger.error?
        # TODO: If you are archiving X, F, J automatically, can this function
        # just raise an exception to indicate issues rather than return a flag?
        # If so, could you implement a simpler log_and_abort() helper function
        # and use that consistently throughout?
        log("Your residual is not m-dimensional.")
        return X, F, J, flag, xkin

    if J0.shape[0] != n or J0.shape[1] != m:
        # TODO: Should this use logger.warn or logger.error?
        log("Your Jacobian is not n by m.")
        X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, -1)
        return X, F, J, flag, xkin

    center_raw_F = F0
    center_raw_J = J0
    center_raw_Info = eval_info
    center_fpr_mask = _fpr_mask_for_x(X[nf])
    F[nf], J[nf], Info[nf] = _apply_fpr_reduction(
        center_raw_F, center_raw_J, center_raw_Info, center_fpr_mask
    )
    center_info = Info[nf]

    if np.any(np.isnan(F[nf])):
        # TODO: Should this use logger.warn or logger.error?
        log("The initial evaluation of F contained a NaN.")
        X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, -3)
        return X, F, J, flag, xkin
    
    Fs[nf] = _call_hfun(F[nf], Info[nf])
    FullFs[nf] = _call_hfun(center_raw_F, center_raw_Info)
    _record_progress(
        phase="initial",
        nf_value=nf,
        x_index=nf,
        active_mask=center_fpr_mask,
        active_objective=Fs[nf],
        full_objective=FullFs[nf],
        incumbent_index=xkin,
        incumbent_active_objective=Fs[xkin],
        incumbent_full_objective=FullFs[xkin],
        delta_value=delta,
        accepted=True,
    )

    log("Initial point evaluated.")
    if fpr_reduction is not None:
        selected = int(np.sum(center_fpr_mask)) if center_fpr_mask is not None else m
        union = int(np.sum(fpr_union_mask)) if fpr_union_mask is not None else selected
        log(
            f"nf: {nf}, full f(x): {FullFs[nf]}, FPR f(x): {Fs[nf]}, "
            f"selected: {selected}/{m}, union: {union}/{m}"
        )
    else:
        log(f"nf: {nf}, f(x): {Fs[nf]}")

    # if we had previous evaluations (an nfs ~=0), we would put them in X, F here
    for i in range(nf + 1):
        Fs[i] = _call_hfun(F[i], Info[i])
    Res = np.zeros(np.shape(F))
    # The original code allocated Hres = np.zeros((n, n, m)), but this branch
    # never updates Hres. Store just its logical dimensions and let
    # combine_leastsquares treat the residual Hessian as implicit zeros.
    Hres = (n, m)
    ng = np.nan  # Needed for early termination, e.g., if a model is never built
    outer_iter = 0  # Counts trust-region iterations; passed to the adaptive-shot hook.
    previous_rho = None  # rho_k is exposed to the callback at the start of iteration k+1.

    while nf + 1 < nfmax:
        center_fpr_mask = _fpr_mask_for_x(X[xkin])
        if center_fpr_mask is not None:
            F[xkin], J[xkin], Info[xkin] = _apply_fpr_reduction(
                center_raw_F, center_raw_J, center_raw_Info, center_fpr_mask
            )
            Fs[xkin] = _call_hfun(F[xkin], Info[xkin])
        FullFs[xkin] = _call_hfun(center_raw_F, center_raw_Info)

        # --- Adaptive-shot hook -------------------------------------------
        # Give an external callback the chance to spend additional measurement
        # shots at the current incumbent BEFORE this iteration's model is built.
        # All heavy logic (FPR-aware D-optimal allocation, the N_k schedule, and
        # updating the dataset that fun() reads) lives OUTSIDE POUNDERS; here we
        # only (a) hand the callback the current state and (b) refresh the center
        # on the updated dataset so the frozen objective used for the rest of
        # this iteration reflects the newly collected shots.  Because the hook
        # fires once per iteration (at the top), the data stays frozen across the
        # incumbent- and trial-point evaluations below, which keeps rho well
        # defined.  iter_callback is None => existing behavior is unchanged.
        if iter_callback is not None:
            cb_state = {
                "x": np.asarray(X[xkin], dtype=float).copy(),
                "delta": float(delta),
                "ng": float(ng),  # previous iteration's model-gradient norm (nan on first pass)
                "iteration": int(outer_iter),
                "nf": int(nf),
                "xkin": int(xkin),
                "fpr_mask": None if center_fpr_mask is None else center_fpr_mask.copy(),
                "previous_rho": previous_rho,
            }
            try:
                cb_result = iter_callback(cb_state)
            except Exception as exc:  # never let shot bookkeeping crash the solve
                log(f"Adaptive-shot callback raised {exc!r}; continuing without new shots.")
                cb_result = None
            if isinstance(cb_result, dict) and cb_result.get("terminate"):
                reason = cb_result.get(
                    "termination_reason", "adaptive callback requested termination"
                )
                log(f"Terminating because {reason}.")
                last_run_metadata["termination_reason"] = str(reason)
                X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, -6)
                return X, F, J, flag, xkin
            if isinstance(cb_result, dict) and cb_result.get("data_changed"):
                # The dataset behind fun() changed; re-evaluate the center so the
                # residuals/weights match the frozen dataset used this iteration.
                center_raw_F, center_raw_J, center_raw_Info = _evaluate_fun(X[xkin])
                F[xkin], J[xkin], Info[xkin] = _apply_fpr_reduction(
                    center_raw_F, center_raw_J, center_raw_Info, center_fpr_mask
                )
                Fs[xkin] = _call_hfun(F[xkin], Info[xkin])
                FullFs[xkin] = _call_hfun(center_raw_F, center_raw_Info)
                log(
                    f"Adaptive-shot hook (iter {outer_iter}): added "
                    f"{cb_result.get('shots_added', '?')} shots; center refreshed."
                )
        outer_iter += 1
        # ------------------------------------------------------------------

        #  1a. Compute the "interpolation set".
        Res[xkin] = F[xkin]
        Gres = J[xkin]

        #  1b. Update the quadratic model
        Cres = F[xkin]
        #Hres = Hres + Hresdel
        t_model = time.perf_counter()
        G, H = _call_combinemodels(Cres, Gres, Hres, Info[xkin])
        log_debug(f"Model combine took {time.perf_counter() - t_model:.2f} seconds.", 0)
        if np.shape(G) == np.shape(Gres):
            # Some notebook sessions may still hold a stale combiner that
            # returns the residual Jacobian instead of the scalar objective
            # gradient. Recover the least-squares gradient directly.
            G = 2 * Gres @ np.asarray(Cres, dtype=float).reshape(-1)
        G = np.asarray(G, dtype=float).reshape(-1)
        H = np.asarray(H, dtype=float)
        if G.size != n or H.shape != (n, n):
            log(f"Model has incompatible shapes: G={G.shape}, H={H.shape}, expected G=({n},), H=({n}, {n}).")
            X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, -4)
            return X, F, J, flag, xkin
        ind_Lnotbinding = (X[xkin] > L) * (G.T > 0)
        ind_Unotbinding = (X[xkin] < U) * (G.T < 0)
        ng = np.linalg.norm(G * (ind_Lnotbinding + ind_Unotbinding).T, 2)

        if fpr_reduction is not None:
            selected = int(np.sum(center_fpr_mask)) if center_fpr_mask is not None else m
            union = int(np.sum(fpr_union_mask)) if fpr_union_mask is not None else selected
            log(
                f"nf: {nf}, delta: {delta}, full f(x): {FullFs[xkin]}, "
                f"FPR f(x): {Fs[xkin]}, selected: {selected}/{m}, "
                f"union: {union}/{m}, ng: {ng}"
            )
        else:
            log(f"nf: {nf}, delta: {delta}, f(x): {Fs[xkin]}, ng: {ng}")

        # 2. Critically test invoked if the projected model gradient is small
        if ng < gtol:
            X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, 0)
            return X, F, J, flag, xkin

        # 3. Solve the subproblem min{G.T * s + 0.5 * s.T * H * s : Lows <= s <= Upps }
        Lows = np.maximum(L - X[xkin], -delta * np.ones((np.shape(L))))
        Upps = np.minimum(U - X[xkin], delta * np.ones((np.shape(U))))
        if spsolver == 3:  # PyROL
            log(f"Starting PyROL trust-region solve with n={n}.")
            t_trsp = time.perf_counter()
            #################### debug pyrol ####################
            # os.makedirs("trsp_debug", exist_ok=True)
            # trsp_debug_path = os.path.join(
            #     "trsp_debug", f"trsp_pyrol_inputs_nf{nf}_xkin{xkin}.npz"
            # )
            # np.savez(
            #     trsp_debug_path,
            #     G=G,
            #     H=H,
            #     Lows=Lows,
            #     Upps=Upps,
            #     n=n,
            #     nf=nf,
            #     xkin=xkin,
            #     delta=delta,
            # )
            # log(f"Saved PyROL trust-region inputs to {trsp_debug_path}.")
            #####################################################
            Xsp, mdec = _solve_trsp_pyrol(G, H, Lows, Upps, n)
            log(f"PyROL trust-region solve took {time.perf_counter() - t_trsp:.2f} seconds.")
        Xsp = Xsp.squeeze()
        step_norm = np.linalg.norm(Xsp, np.inf)

        # 4. Evaluate the function at the new point
        if mdec != 0:
            Xsp = np.minimum(U, np.maximum(L, X[xkin] + Xsp))  # Temp safeguard; note Xsp is not a step anymore

            # Project if we're within machine precision
            for i in range(n):  # This will need to be cleaned up eventually
                if (U[i] - Xsp[i] < eps * abs(U[i])) and (U[i] > Xsp[i] and G[i] >= 0):
                    Xsp[i] = U[i]
                    log_debug("eps project!", 0)
                elif (Xsp[i] - L[i] < eps * abs(L[i])) and (L[i] < Xsp[i] and G[i] >= 0):
                    Xsp[i] = L[i]
                    log_debug("eps project!", 0)

            nf += 1
            X[nf] = Xsp
            t_eval = time.perf_counter()
            trial_F, trial_J, eval_info = _evaluate_fun(X[nf])
            trial_raw_F = np.asarray(trial_F, dtype=float).reshape(-1)
            trial_raw_J = np.asarray(trial_J, dtype=float)
            trial_raw_Info = eval_info
            F[nf], J[nf], Info[nf] = _apply_fpr_reduction(
                trial_raw_F, trial_raw_J, trial_raw_Info, center_fpr_mask
            )
            eval_info = Info[nf]
            log(f"Residual/Jacobian evaluation took {time.perf_counter() - t_eval:.2f} seconds.")
            if np.any(np.isnan(F[nf])):
                X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, -3)
                return X, F, J, flag, xkin
            Fs[nf] = _call_hfun(F[nf], Info[nf])
            FullFs[nf] = _call_hfun(trial_raw_F, trial_raw_Info)

            rho_active = (Fs[nf] - Fs[xkin]) / mdec
            rho_full = (FullFs[nf] - FullFs[xkin]) / mdec
            rho = rho_full if rho_uses_full_objective else rho_active
            previous_rho = float(rho)
            previous_xkin = xkin

            # 4a. Update the center
            if rho > 0:
                # Update model to reflect new center
                xkin = nf  # Change current center
                J[previous_xkin] = None
                center_raw_F = trial_raw_F
                center_raw_J = trial_raw_J
                center_raw_Info = trial_raw_Info
                center_info = Info[nf]
            else:
                J[nf] = None
                trial_raw_F = None
                trial_raw_J = None
                trial_raw_Info = None

            _record_progress(
                phase="trial",
                nf_value=nf,
                x_index=nf,
                active_mask=center_fpr_mask,
                active_objective=Fs[nf],
                full_objective=FullFs[nf],
                incumbent_index=xkin,
                incumbent_active_objective=Fs[xkin],
                incumbent_full_objective=FullFs[xkin],
                delta_value=delta,
                ng_value=ng,
                rho_value=rho,
                rho_active_value=rho_active,
                rho_full_value=rho_full,
                mdec_value=mdec,
                step_norm_value=step_norm,
                accepted=bool(rho > 0),
            )

            # 4b. Update the trust-region radius:
            if (rho >= eta1) and (step_norm > 0.75 * delta):
                delta = min(delta * gam1, maxdelta)
            else:
                delta = max(delta * gam0, mindelta)
        else:
            # TODO: Should this use logger.warn or logger.error?
            log("Model decrease cannot be found, terminating. ")
            X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, -2)
            return X, F, J, flag, xkin

    # TODO: Should this use logger.warn or logger.error?
    # TODO: If you implement a log_and_abort(), should this particular instance
    # use that or just log a warning and return all arguments as it does for
    # success?
    log("Number of function evals exceeded")
    flag = ng
    return X, F, J, flag, xkin
