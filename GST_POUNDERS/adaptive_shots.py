"""Adaptive measurement-shot allocation for GST + POUNDERS.

This module implements the D-optimal (and A-optimal) shot-allocation machinery
described in Sections 8-10 of the paper *Stochastic Nonlinear Least Squares*
(``overleaf/Stochastic_Nonlinear_Least_Squares/main.tex``), plus the per-iteration
shot-budget schedule N_k discussed with Jeff and Matt.

Design intent (per Jeff): these are **plain functions** meant to be called from
the outside of the POUNDERS loop.  Nothing here mutates optimizer state or imports
pyGSTi/POUNDERS.  The caller passes in the current Jacobian ``J`` (rows indexed by
circuit-outcome pair, columns by model parameter), the per-experiment single-shot
variances ``sigma2``, and the shots already spent ``n``; it gets back an integer
vector of *additional* shots to spend.

Notation map (paper  ->  code)
------------------------------
    i = (s, beta)          row index of the FPR-selected residual set  -> axis 0 of J
    J_i                    i-th row of the Jacobian                    -> J[i]
    sigma_i^2              single-shot variance of experiment i        -> sigma2[i]
    n_i                    shots already spent on experiment i         -> n[i]
    W = diag(n_i/sigma_i^2)  current weights                           -> weights(n, sigma2)
    H = J^T W J            current Fisher information (eq. H=J^T W J)   -> fisher_information(...)
    rho_i >= 0             *additional* shots on experiment i          -> the returned vector
    N                      additional shot budget for this iteration   -> the schedule N_k

Single-shot variance for a GST outcome probability p_i is the Bernoulli variance
sigma_i^2 = p_i (1 - p_i); pass ``sigma2 = p*(1-p)`` (clipped), matching the
``var_p = p*(1-p)/shots`` convention in ``GST_model.ipynb``.

Everything is pure NumPy so it runs in the plain environment (no cvxpy/pyGSTi).
Run ``python adaptive_shots.py`` for a self-test.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bernoulli_single_shot_variance",
    "fisher_information",
    "stable_inverse",
    "allocate_shots",
    "allocate_shots_per_circuit",
    "infidelity_metric_hessian",
    "geometric_schedule",
    "adaptive_schedule",
]

# ---------------------------------------------------------------------------
# PER-OUTCOME allocator (previous version).  Design variable per (circuit,outcome)
# row with weight 1/sigma^2 (Bernoulli); the hook then max-aggregates to circuits.
# Physically approximate, but empirically stronger on GST infidelity than the exact
# per-circuit block (it under-spends -> less over-concentration).  Kept alongside
# allocate_shots_per_circuit so either can be selected.
# ---------------------------------------------------------------------------


def bernoulli_single_shot_variance(p, floor=1e-12):
    """Single-shot Bernoulli variance sigma^2 = p(1-p), floored."""
    p = np.asarray(p, dtype=float)
    return np.maximum(p * (1.0 - p), floor)


def weights(n, sigma2):
    """Diagonal of W = diag(n_i / sigma_i^2)."""
    return np.asarray(n, dtype=float) / np.asarray(sigma2, dtype=float)


def fisher_information(J, w, ridge=0.0):
    """H = J^T diag(w) J (+ ridge I)."""
    J = np.asarray(J, dtype=float); w = np.asarray(w, dtype=float)
    H = (J * w[:, None]).T @ J
    if ridge:
        H = H + ridge * np.eye(H.shape[0])
    return H


# Directions whose eigenvalue is below RANK_CUTOFF * lambda_max are treated as
# unidentifiable and dropped.  GST's gauge freedom makes H exactly rank-deficient,
# and 1/p weights up to ~1e5 push the identifiable spectrum over ~1e9, so H as
# assembled is indefinite at the 1e-7 level from roundoff alone.  1e-10 sits well
# below the smallest genuinely-informative eigenvalue and well above the gauge
# block.  allocate_shots_per_circuit reports the surviving rank as info["rank"],
# so a cutoff that is eating real directions shows up there.
RANK_CUTOFF = 1e-10


def stable_inverse(H, cutoff=RANK_CUTOFF, with_logdet=False):
    """Pseudo-inverse of H restricted to its identifiable (non-gauge) subspace.

    ``np.linalg.inv(H + ridge*I)`` is not usable here.  H is rank-deficient by the
    gauge dimension, so a flat ridge leaves those directions at ~ridge and the
    inverse blows them up by 1/ridge.  D-optimality survives that (its score
    ``J_i H^-1 J_i^T`` applies H^-1 once, and J_i is orthogonal to the null space
    in exact arithmetic), but A- and L-optimality apply H^-1 *twice* and so square
    an already ~1e16 condition number -- their circuit rankings decorrelate from
    the true ones entirely.

    Symmetrizing and truncating at a relative cutoff restricts every criterion to
    the identifiable subspace, which is what D/A/L are meant to score over anyway
    (this is Ds-optimality treating the gauge block as nuisance parameters).

    Returns Hinv, or (Hinv, pseudo_logdet, rank) when ``with_logdet``.  The
    pseudo-determinant is the product of the KEPT eigenvalues only -- using
    slogdet(H) instead would add a constant ~rank_deficiency*log(ridge) and go to
    -inf once the ridge is removed.
    """
    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)
    ev, V = np.linalg.eigh(H)
    ev_max = ev[-1]
    if not np.isfinite(ev_max) or ev_max <= 0.0:
        # Degenerate design (no information at all); fall back to zeros so the
        # caller's argmax stays well-defined instead of raising.
        Hinv = np.zeros_like(H)
        return (Hinv, -np.inf, 0) if with_logdet else Hinv
    keep = ev > cutoff * ev_max
    Vk, evk = V[:, keep], ev[keep]
    Hinv = (Vk / evk) @ Vk.T
    if with_logdet:
        return Hinv, float(np.sum(np.log(evk))), int(keep.sum())
    return Hinv


def _kept_eigenvalues(H, cutoff=RANK_CUTOFF):
    """Eigenvalues of sym(H) above the relative cutoff, via eigvalsh (no vectors)."""
    H = np.asarray(H, dtype=float)
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    ev_max = ev[-1]
    if not np.isfinite(ev_max) or ev_max <= 0.0:
        return None
    return ev[ev > cutoff * ev_max]


def stable_pseudo_logdet(H, cutoff=RANK_CUTOFF):
    """log-pseudo-determinant of H over its identifiable subspace.

    Same truncation as ``stable_inverse`` but via ``eigvalsh`` (no eigenvectors),
    which is several times cheaper.  The D-optimality objective is evaluated once
    per Frank-Wolfe line-search trial, so it must not build a full inverse.
    """
    evk = _kept_eigenvalues(H, cutoff)
    return -np.inf if evk is None else float(np.sum(np.log(evk)))


def stable_pseudo_trace_inv(H, cutoff=RANK_CUTOFF):
    """Tr(H^+) over the identifiable subspace -- the A-optimality objective.

    Trace is basis-independent, so this needs eigenvalues only; same eigvalsh
    shortcut as the D objective.  L-optimality cannot use it (Tr(M H^+) depends
    on how M sits relative to H's eigenbasis, so it needs the eigenvectors).
    """
    evk = _kept_eigenvalues(H, cutoff)
    return np.inf if evk is None else float(np.sum(1.0 / evk))


def row_quadratic_forms(J, Hinv):
    """q_i = J_i Hinv J_i^T for every row, vectorized."""
    J = np.asarray(J, dtype=float)
    return np.einsum("ij,ji->i", J, Hinv @ J.T)


def dopt_scores(J, Hinv, inv_sigma2):
    """D-optimal marginal gains (1/sigma_i^2) J_i H^{-1} J_i^T."""
    return inv_sigma2 * row_quadratic_forms(J, Hinv)


def aopt_scores(J, Hinv, inv_sigma2):
    """A-optimal marginal gains (1/sigma_i^2) ||H^{-1} J_i^T||^2."""
    J = np.asarray(J, dtype=float); Y = Hinv @ J.T
    return inv_sigma2 * np.sum(Y * Y, axis=0)


def lopt_scores(J, Hinv, inv_sigma2, M):
    """L-optimal marginal gains (1/sigma_i^2) J_i (H^{-1} M H^{-1}) J_i^T.

    This is the marginal decrease in Tr(M H^{-1}) from one more shot on row i.
    With M = I it is exactly A-optimality; with M = the infidelity Hessian,
    Tr(M H^{-1}) is (to leading order) the *expected mean-gate infidelity itself*,
    so L-optimality allocates shots to directly minimize infidelity rather than a
    volume (D) or unweighted-variance (A) surrogate.
    """
    G = Hinv @ np.asarray(M, dtype=float) @ Hinv
    return inv_sigma2 * row_quadratic_forms(J, G)


_SCORE_FUNCS = {"D": dopt_scores, "A": aopt_scores}


def _make_score_func(criterion, metric_M=None):
    """Return a score_func(J, Hinv, inv_sigma2) closure for the chosen criterion."""
    if criterion in _SCORE_FUNCS:
        return _SCORE_FUNCS[criterion]
    if criterion == "L":
        if metric_M is None:
            raise ValueError("criterion='L' requires metric_M (e.g. the infidelity "
                             "Hessian from infidelity_metric_hessian()).")
        M = np.asarray(metric_M, dtype=float)
        return lambda J, Hinv, inv_sigma2: lopt_scores(J, Hinv, inv_sigma2, M)
    raise ValueError(f"unknown criterion {criterion!r}; use 'D', 'A', or 'L'.")


def frank_wolfe_relaxed(J, sigma2, N, H0, criterion="D", max_iter=200, gap_tol=1e-8,
                        rho0=None, step="line_search", metric_M=None):
    """Continuous relaxation of the per-row design via Frank-Wolfe on the simplex."""
    score_func = _make_score_func(criterion, metric_M)
    Mmat = None if metric_M is None else np.asarray(metric_M, dtype=float)
    J = np.asarray(J, dtype=float); sigma2 = np.asarray(sigma2, dtype=float)
    inv_sigma2 = 1.0 / sigma2
    m = J.shape[0]
    rho = np.full(m, N / m) if rho0 is None else np.array(rho0, dtype=float)

    def objective_H(H):
        """Criterion value given an already-assembled H (higher is better)."""
        if criterion == "D":               # pseudo-determinant over kept directions
            return stable_pseudo_logdet(H)
        if criterion == "A":               # minimize Tr(H^{-1}) -> maximize its negative
            return -stable_pseudo_trace_inv(H)
        return -np.trace(Mmat @ stable_inverse(H))   # L-opt needs the eigenvectors

    history = []; gap = np.inf
    # Fisher info is LINEAR in the weight vector, so H along the line-search ray
    # rho + gamma*d is H_rho + gamma*H_d with both endpoints assembled once per FW
    # iteration.  Re-running fisher_information() per trial dominated the runtime.
    H_fisher = fisher_information(J, rho * inv_sigma2)
    obj = objective_H(H0 + H_fisher)
    for l in range(max_iter):
        if l and l % 25 == 0:
            # Resync the incrementally-updated H against an exact rebuild; entries
            # reach ~1e9 here, so accumulating gamma*H_d drifts over many steps.
            H_fisher = fisher_information(J, rho * inv_sigma2)
        H_rho = H0 + H_fisher
        Hinv = stable_inverse(H_rho)
        g = score_func(J, Hinv, inv_sigma2)
        i_star = int(np.argmax(g)); s = np.zeros(m); s[i_star] = N
        gap = float(g @ (s - rho))
        history.append({"iter": l, "objective": obj, "gap": gap})
        if gap <= gap_tol:
            break
        d = s - rho
        # H(s) is rank one: s puts the whole budget N on row i_star.
        a = J[i_star]
        H_s = (N * inv_sigma2[i_star]) * np.outer(a, a)
        H_d = H_s - H_fisher
        if step == "classic":
            gamma = 2.0 / (l + 2.0)
            rho = rho + gamma * d
            H_fisher = H_fisher + gamma * H_d
            obj = objective_H(H0 + H_fisher)
        else:
            gamma = 1.0; base = obj
            while gamma > 1e-10:
                cand_obj = objective_H(H_rho + gamma * H_d)
                if cand_obj > base + 1e-12 * abs(base):
                    rho = rho + gamma * d
                    H_fisher = H_fisher + gamma * H_d
                    obj = cand_obj
                    break
                gamma *= 0.5
            else:
                break
    return rho, {"objective": obj, "gap": gap, "n_iter": len(history), "history": history}


def round_allocation_greedy(rho_star, J, sigma2, N, H0, criterion="D", metric_M=None,
                            refactor_every=50):
    """Floor + greedy completion with Sherman-Morrison rank-1 updates.

    ``refactor_every`` rebuilds Hinv from scratch periodically.  The rank-1
    downdates are valid on the truncated pseudo-inverse (each update direction
    J_i lies in the identifiable subspace, so iterates stay there) but roundoff
    accumulates over hundreds of steps on a spectrum spanning ~1e16.
    """
    score_func = _make_score_func(criterion, metric_M)
    J = np.asarray(J, dtype=float); sigma2 = np.asarray(sigma2, dtype=float)
    inv_sigma2 = 1.0 / sigma2
    m = np.floor(np.asarray(rho_star, dtype=float)).astype(int)
    R = int(round(N - m.sum()))
    Hinv = stable_inverse(H0 + fisher_information(J, m * inv_sigma2))
    for step in range(max(R, 0)):
        g = score_func(J, Hinv, inv_sigma2)
        i_star = int(np.argmax(g)); m[i_star] += 1
        if refactor_every and (step + 1) % refactor_every == 0:
            Hinv = stable_inverse(H0 + fisher_information(J, m * inv_sigma2))
            continue
        a = J[i_star]; alpha = inv_sigma2[i_star]; Hinv_a = Hinv @ a
        denom = 1.0 + alpha * (a @ Hinv_a)
        Hinv = Hinv - (alpha / denom) * np.outer(Hinv_a, Hinv_a)
    return m, {"leftover": R}


def allocate_shots(J, sigma2, N, n=None, H0=None, criterion="D", ridge=1e-9,
                   fw_kwargs=None, metric_M=None):
    """Per-ROW (per outcome) D/A/L-optimal allocation: FW relaxation + integer rounding.

    criterion : {"D", "A", "L"}.  "L" requires metric_M (the infidelity Hessian) and
    allocates to directly minimize Tr(M H^{-1}) ~= expected infidelity.
    """
    J = np.asarray(J, dtype=float); sigma2 = np.asarray(sigma2, dtype=float)
    m, d = J.shape
    if H0 is None:
        w0 = np.zeros(m) if n is None else weights(n, sigma2)
        H0 = fisher_information(J, w0, ridge=ridge)
    elif ridge:
        H0 = H0 + ridge * np.eye(d)
    rho, fw_info = frank_wolfe_relaxed(J, sigma2, N, H0, criterion=criterion,
                                       metric_M=metric_M, **(fw_kwargs or {}))
    extra, round_info = round_allocation_greedy(rho, J, sigma2, N, H0,
                                                criterion=criterion, metric_M=metric_M)
    return extra, {"rho": rho, "fw": fw_info, "rounding": round_info, "H0": H0}


def allocate_shots_per_circuit(J, p, N, circuit_of_row, n_circuit=None, H0=None,
                               criterion="D", ridge=1e-9, prob_floor=1e-9,
                               max_iter=200, gap_tol=1e-8, metric_M=None,
                               refactor_every=50):
    """Per-CIRCUIT D/A-optimal shot allocation with MULTINOMIAL Fisher blocks.

    A single circuit-shot samples all of a circuit's outcomes jointly, so the design
    variable is one integer rho_s per circuit (NOT per outcome).  Each circuit s
    contributes B_s = J_s^T diag(1/p_s) J_s (the per-shot multinomial Fisher info;
    equals J_s^T Sigma_s^+ J_s since the rows sum to zero), and
        H(rho) = H0 + sum_s rho_s B_s.
    D-optimality scores tr(H^{-1} B_s); A-optimality scores tr(H^{-1} B_s H^{-1}).
    Both reduce to per-row quadratic forms weighted by 1/p_row, summed per circuit.

    J : (m, d) Jacobian, rows = (circuit, outcome);  p : (m,) outcome probabilities.
    N : int circuit-shot budget (sum_s rho_s = N, respected EXACTLY).
    circuit_of_row : (m,) int, circuit index (0..n_circuits-1) of each row.
    n_circuit : (n_circuits,) or None, shots already on each circuit -> H0.
    Returns (extra_per_circuit int array summing to N, info dict).
    """
    J = np.asarray(J, dtype=float); p = np.asarray(p, dtype=float)
    circuit_of_row = np.asarray(circuit_of_row, dtype=int).reshape(-1)
    m, d = J.shape
    n_circuits = int(circuit_of_row.max()) + 1 if m else 0
    inv_p = 1.0 / np.maximum(p, prob_floor)
    Mmat = None
    if criterion == "L":
        if metric_M is None:
            raise ValueError("criterion='L' requires metric_M (the infidelity Hessian).")
        Mmat = np.asarray(metric_M, dtype=float)

    def _H_design(shots_c):
        w = np.asarray(shots_c, dtype=float)[circuit_of_row] * inv_p
        return (J * w[:, None]).T @ J

    if H0 is None:
        n0 = np.zeros(n_circuits) if n_circuit is None else np.asarray(n_circuit, dtype=float)
        H0 = _H_design(n0) + ridge * np.eye(d)
    elif ridge:
        H0 = np.asarray(H0, dtype=float) + ridge * np.eye(d)

    def _scores(Hinv):
        if criterion == "L":                       # tr(H^{-1} M H^{-1} B_s), G = H^{-1} M H^{-1}
            G = Hinv @ Mmat @ Hinv
            rowq = np.einsum("ij,ji->i", J, G @ J.T)
        else:
            Y = Hinv @ J.T
            rowq = np.einsum("ij,ji->i", J, Y) if criterion == "D" else np.sum(Y * Y, axis=0)
        rs = rowq * inv_p
        sc = np.zeros(n_circuits); np.add.at(sc, circuit_of_row, rs)
        return sc

    # Frank-Wolfe over the circuit simplex {rho >= 0, sum rho_s = N}
    rho = np.full(n_circuits, N / n_circuits)
    gap = np.inf
    rank = d
    for _l in range(max_iter):
        Hinv, _logdet, rank = stable_inverse(H0 + _H_design(rho), with_logdet=True)
        g = _scores(Hinv)
        i_star = int(np.argmax(g))
        gap = float(N * g[i_star] - g @ rho)
        if gap <= gap_tol:
            break
        gamma = 2.0 / (_l + 2.0)
        rho *= (1.0 - gamma); rho[i_star] += gamma * N

    # floor + greedy completion (R = N - sum(floor) < n_circuits); block Woodbury updates
    m_int = np.floor(rho).astype(int)
    R = int(round(N - m_int.sum()))
    Hinv = stable_inverse(H0 + _H_design(m_int.astype(float)))
    for step in range(max(R, 0)):
        g = _scores(Hinv)
        i_star = int(np.argmax(g))
        m_int[i_star] += 1
        if refactor_every and (step + 1) % refactor_every == 0:
            Hinv = stable_inverse(H0 + _H_design(m_int.astype(float)))
            continue
        rows = np.where(circuit_of_row == i_star)[0]
        Js = J[rows]                              # (k, d)
        HJt = Hinv @ Js.T                         # (d, k)
        Mw = np.diag(p[rows]) + Js @ HJt          # (k, k): diag(1/inv_p) + Js Hinv Js^T
        Hinv = Hinv - HJt @ np.linalg.solve(Mw, HJt.T)
    return m_int, {"rho": rho, "gap": gap, "R": R, "rank": int(rank), "d": int(d)}



# ---------------------------------------------------------------------------
# Infidelity metric M for L-optimality  (M = d^2 infidelity / d theta^2)
# ---------------------------------------------------------------------------


def infidelity_metric_hessian(infid_fn, x, eps=1e-3, diagonal_only=False, psd_floor=0.0):
    """Finite-difference Hessian M of a scalar infidelity callable, for L-optimality.

    Parameters
    ----------
    infid_fn : callable(theta) -> float
        A (gauge-fixed) mean-gate infidelity of ``model(theta)`` relative to a FIXED
        reference model(``x``).  Because ``infid_fn(x) ~= 0`` and ``x`` is a local
        minimum, ``M = d^2 infid / d theta^2 |_x`` is PSD.  Then, to leading order,
        ``E[infidelity of the estimate] ~= Tr(M H^{-1})`` -- exactly what L-optimality
        (``criterion='L'``) minimizes.  (In simulation you may build the reference at
        the *true* parameters for a best-case metric; in a real experiment use the
        current incumbent as the reference and rebuild occasionally.)
    x : (d,) array          point at which to evaluate the Hessian (the reference).
    eps : float             central-difference step (balances truncation vs any
                            gauge-opt noise in ``infid_fn``; 1e-3 is a good default).
    diagonal_only : bool    if True, skip the O(d^2) off-diagonal terms (1 + 2d evals
                            instead of ~2 d^2); a cheap approximation for large d.
    psd_floor : float or None   clip eigenvalues up to >= psd_floor (>=0 keeps M PSD).

    Notes
    -----
    Each ``infid_fn`` call includes a gauge optimization for GST, so build M ONCE and
    cache it rather than rebuilding every POUNDERS iteration.
    """
    x = np.asarray(x, dtype=float); d = x.size
    f0 = float(infid_fn(x))
    e = np.eye(d) * eps
    fp = np.array([float(infid_fn(x + e[i])) for i in range(d)])
    fm = np.array([float(infid_fn(x - e[i])) for i in range(d)])
    H = np.zeros((d, d))
    np.fill_diagonal(H, (fp - 2.0 * f0 + fm) / (eps * eps))
    if not diagonal_only:
        for i in range(d):
            for j in range(i + 1, d):
                fpp = float(infid_fn(x + e[i] + e[j]))
                fpm = float(infid_fn(x + e[i] - e[j]))
                fmp = float(infid_fn(x - e[i] + e[j]))
                fmm = float(infid_fn(x - e[i] - e[j]))
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
    H = 0.5 * (H + H.T)
    if psd_floor is not None:
        w, V = np.linalg.eigh(H)
        H = (V * np.maximum(w, psd_floor)) @ V.T
    return H


# ---------------------------------------------------------------------------
# Shot-budget schedules N_k  (how many *new* shots to spend in iteration k)
# ---------------------------------------------------------------------------


def geometric_schedule(k, base=1.5, n0=32, n_min=None, n_max=8192):
    """Geometric per-iteration budget N_k = clip(n0 * base**k, n_min, n_max).

    Jeff's first-pass schedule: N_k should grow (roughly) geometrically with a
    base *smaller than 2*.  The rationale: early iterations should not waste
    precious shots on circuits that FPR will likely drop once x settles, whereas
    late iterations -- once the FPR-selected set has stabilized -- should reach
    high precision.  A cap n_max prevents runaway budgets.

    Parameters
    ----------
    k : int          zero-based POUNDERS iteration index.
    base : float     growth base, 1 < base < 2 (default 1.5).
    n0 : int         budget at iteration 0.
    n_min, n_max : int   clamp on the returned budget (n_min defaults to n0).
    """
    # Jeff's guideline is 1 < base < 2 (grow, but slower than doubling); not enforced.
    n_min = n0 if n_min is None else n_min
    val = n0 * (base ** k)
    return int(np.clip(round(val), n_min, n_max))


def adaptive_schedule(
    grad_norm=None,
    obj_value=None,
    scale=1.0,
    n_min=32,
    n_max=8192,
    eps=1e-8,
):
    """Progress-adaptive budget: spend more shots as the optimizer nears a solution.

    Matt's idea: let N_k grow like 1 / ||model gradient|| or 1 / objective value,
    both capped.  As the optimizer converges (gradient norm and objective shrink),
    the requested precision -- and thus the shot budget -- rises automatically,
    concentrating shots when the FPR-selected set has settled.

    Exactly one of ``grad_norm`` or ``obj_value`` should be supplied.  Returns
    clip(scale / max(quantity, eps), n_min, n_max), rounded to an int.
    """
    if (grad_norm is None) == (obj_value is None):
        raise ValueError("supply exactly one of grad_norm or obj_value.")
    quantity = grad_norm if grad_norm is not None else obj_value
    val = scale / max(float(quantity), eps)
    return int(np.clip(round(val), n_min, n_max))


# ---------------------------------------------------------------------------
# Self-test: run `python adaptive_shots.py`
# ---------------------------------------------------------------------------


def _selftest():
    rng = np.random.default_rng(0)
    nc, k, d = 40, 2, 8               # 40 circuits, 2 outcomes each, 8 parameters
    m = nc * k
    circuit_of_row = np.repeat(np.arange(nc), k)
    J0 = rng.standard_normal((nc, d))
    J = np.empty((m, d)); J[0::k] = J0; J[1::k] = -J0      # anti-correlated outcomes
    p0 = rng.uniform(0.1, 0.9, nc)
    p = np.empty(m); p[0::k] = p0; p[1::k] = 1.0 - p0
    n_circuit = np.full(nc, 100.0)
    N = 500
    inv_p = 1.0 / np.maximum(p, 1e-9)
    H0 = (J * (n_circuit[circuit_of_row] * inv_p)[:, None]).T @ J + 1e-9 * np.eye(d)

    # a fixed PSD "infidelity" metric M for the L-optimality test
    A = rng.standard_normal((d, d)); M = A @ A.T + 0.1 * np.eye(d)

    def _H(alloc):
        return H0 + (J * (alloc[circuit_of_row] * inv_p)[:, None]).T @ J

    def _ld_tr(alloc):
        Hi = np.linalg.inv(_H(alloc))
        return np.linalg.slogdet(_H(alloc))[1], np.trace(Hi)

    def _trM(alloc):
        return float(np.trace(M @ np.linalg.inv(_H(alloc))))

    for crit in ("D", "A", "L"):
        extra, info = allocate_shots_per_circuit(J, p, N, circuit_of_row,
                                                 n_circuit=n_circuit, criterion=crit,
                                                 metric_M=(M if crit == "L" else None))
        assert extra.sum() == N, f"budget not met: {extra.sum()} != {N}"
        assert np.all(extra >= 0)
        uni = np.full(nc, N / nc)
        if crit == "D":
            better = _ld_tr(extra.astype(float))[0] >= _ld_tr(uni)[0]; metric = "log det H"
        elif crit == "A":
            better = _ld_tr(extra.astype(float))[1] <= _ld_tr(uni)[1]; metric = "Tr H^{-1}"
        else:
            better = _trM(extra.astype(float)) <= _trM(uni); metric = "Tr(M H^{-1})"
        print(f"[{crit}-opt] budget met ({extra.sum()}), gap={info['gap']:.2e}, "
              f"{metric} beats uniform: {better}")
        assert better, f"{crit}-optimal design did not beat uniform on {metric}"

    # per-OUTCOME allocator (the one the hook uses) with L must also beat uniform on Tr(M H^{-1})
    sigma2 = np.maximum(p * (1.0 - p), 1e-12)
    exL, _ = allocate_shots(J, sigma2, N, n=np.repeat(n_circuit, k), criterion="L", metric_M=M)
    assert exL.sum() == N and np.all(exL >= 0)
    # map per-row allocation to a per-circuit count via max-aggregate (as the hook does)
    perc = np.zeros(nc); np.maximum.at(perc, circuit_of_row, exL)
    print(f"[L-opt per-outcome] rows placed {exL.sum()}, Tr(M H^-1) beats uniform: "
          f"{_trM(perc) <= _trM(np.full(nc, perc.sum() / nc))}")

    print("geometric schedule N_k (base=1.5):",
          [geometric_schedule(k, base=1.5, n0=32) for k in range(8)])
    print("adaptive schedule (1/grad):",
          [adaptive_schedule(grad_norm=g, scale=1e3) for g in (10, 3, 1, 0.3, 0.1)])
    print("self-test passed.")


if __name__ == "__main__":
    _selftest()
