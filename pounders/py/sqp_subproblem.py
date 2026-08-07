"""
Active-set TR-SQP subproblem solver  (open-source version using `qpsolvers`).

ROBUST SOLVE: badly-scaled problems (e.g. HS84, gradients ~1e6) can defeat any
single interior-point solver. We try a primary solver and fall through a list of
backups before giving up, so a numerical failure in one solver does not abort the
whole optimization.
"""

import numpy as np
from qpsolvers import solve_qp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Tried in order; missing or failing solvers are skipped, not fatal.
_SOLVER_FALLBACKS = ["proxqp", "osqp", "scs", "highs", "quadprog", "cvxopt", "ecos"]


def _solve_qp_robust(P, q, G, h, lb, ub, solver):
    """Try `solver` first, then the fallback list. Returns the solution vector or
    None if every available solver failed."""
    order, seen, x = [solver] + _SOLVER_FALLBACKS, set(), None
    tried = []
    for slv in order:
        if slv in seen:
            continue
        seen.add(slv); tried.append(slv)
        try:
            x = solve_qp(P, q, G=G, h=h, lb=lb, ub=ub, solver=slv)
        except Exception:
            x = None                      # solver not installed or errored
        if x is not None and np.all(np.isfinite(x)):
            return x, slv
    return None, tried


def _backup_step(g_f, H, c_active, G_active, lb, ub, n):
    """Steepest-descent BISECTION backup, used only when every QP solver fails
    (Matt: 'you generally have to do something when a subproblem fails').

    Direction d = -g_f (descent on the objective model). Scale d so the full
    step reaches the box boundary, then bisect the length to the largest alpha
    keeping the LINEARIZED active constraints feasible (c_i + g_ci^T s <= 0) and
    s inside the box. Returns (s, tau, pred) with the SAME meaning as the QP:
        tau  = max{ g_f^T s, max_i (c_i + g_ci^T s) }
        pred = -g_f^T s                   (= m^f(x_k) - m^f(x_k+s), linear model)
    NOTE: this is a suboptimal fallback, so the resulting tau (hence chi) is a
    conservative under-estimate of the true LP value -- acceptable for a last
    resort, but worth knowing when it fires."""
    d = -np.asarray(g_f, float)
    s = np.zeros(n)
    if not np.all(np.isfinite(d)) or np.linalg.norm(d) == 0:
        return s, 0.0, 0.0

    # largest t with lb <= t*d <= ub  (step to the box boundary)
    t_max = np.inf
    for j in range(n):
        if d[j] > 0:
            t_max = min(t_max, ub[j] / d[j])
        elif d[j] < 0:
            t_max = min(t_max, lb[j] / d[j])
    if not np.isfinite(t_max) or t_max <= 0:
        t_max = 1.0

    def feasible(alpha):
        ss = alpha * d
        if np.any(ss < lb - 1e-12) or np.any(ss > ub + 1e-12):
            return False
        for i in range(len(c_active)):
            if c_active[i] + float(G_active[i] @ ss) > 1e-12:
                return False
        return True

    if feasible(t_max):
        alpha = t_max
    else:
        lo, hi = 0.0, t_max
        for _ in range(60):                 # bisection on the step length
            mid = 0.5 * (lo + hi)
            if feasible(mid):
                lo = mid
            else:
                hi = mid
        alpha = lo

    s = alpha * d
    vals = [float(g_f @ s)] + [float(c_active[i] + G_active[i] @ s)
                               for i in range(len(c_active))]
    tau = max(vals)
    pred = -float(g_f @ s)        # LINEAR model decrease m^f(x_k)-m^f(x_k+s)
    return s, tau, pred


def solve_subproblem(x_k, g_f, H, c_active, G_active, delta, low, upp, solver="clarabel"):
    n = len(x_k)
    n_active = len(c_active)
    N = n + 1  # variables: [s (n), tau (1)]

    # Objective: min tau + 0.5 * s^T H s
    P = np.zeros((N, N))
    P[:n, :n] = H
    q = np.zeros(N)
    q[n] = 1.0

    # Inequality: [g_f^T s - tau <= 0] and [g_ci^T s - tau <= -c_i(x_k)]
    G_ineq = np.zeros((1 + n_active, N))
    h_ineq = np.zeros(1 + n_active)
    G_ineq[0, :n] = g_f
    G_ineq[0, n]  = -1.0          # g_f^T s - tau <= 0
    for i in range(n_active):
        G_ineq[1 + i, :n] = G_active[i]
        G_ineq[1 + i, n]  = -1.0  # g_ci^T s - tau <= -c_i(x_k)
        h_ineq[1 + i]     = -c_active[i]

    # Box bounds on [s; tau]: s in [-delta, delta] intersected with [low-x_k, upp-x_k]
    lb = np.empty(N)
    ub = np.empty(N)
    lb[:n] = np.maximum(-delta, low - x_k)
    ub[:n] = np.minimum( delta, upp - x_k)
    lb[n], ub[n] = -np.inf, np.inf

    x, used = _solve_qp_robust(P, q, G_ineq, h_ineq, lb, ub, solver)
    if x is not None:
        s_val   = x[:n]
        tau_val = float(x[n])
        # Predicted decrease of the LINEAR model (Alg 2): m^f is f(x_k)+g_f^T s,
        # so m^f(x_k) - m^f(x_k+s) = -g_f^T s.  H is only the subproblem
        # regularizer; it does NOT enter the model, hence not the predicted
        # decrease used for rho.
        pred_decrease = -float(g_f @ s_val)
        return s_val, tau_val, pred_decrease

    # Every QP solver failed -> steepest-descent bisection backup (per Matt).
    return _backup_step(np.asarray(g_f, float), np.asarray(H, float),
                        np.asarray(c_active, float), np.asarray(G_active, float),
                        lb[:n], ub[:n], n)