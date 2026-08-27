"""Wire the adaptive-shot allocator into gradient_pounders via a per-iteration hook.

`gradient_pounders.pouders(..., iter_callback=hook)` calls `hook(state)` once at the
top of every trust-region iteration, at the current incumbent.  This module builds
that `hook` from the notebook's own problem-specific functions (dependency
injection), so nothing here imports pyGSTi or the notebook -- the notebook passes
its callables in.  All the design math lives in ``adaptive_shots.py``; this file is
just the glue that turns "we are at iterate x_k with radius Delta_k" into "collect
these extra shots."

The state dict POUNDERS hands the hook (see gradient_pounders.py):
    state = {"x", "delta", "ng", "iteration", "nf", "xkin", "fpr_mask",
             "previous_rho", and optional cached center probabilities/Jacobian}
The hook must return either None / {"data_changed": False} (nothing collected) or
{"data_changed": True, "shots_added": <int>} so POUNDERS knows to re-evaluate the
center on the updated dataset.

Injected callables (all problem-specific, supplied by the notebook):
    probability_and_jacobian(x, circuits) -> (p, J)
        p : (r,) model probabilities for the residual rows of `circuits`
        J : (r, d) UNWEIGHTED d p / d x  (NOT divided by sigma)
    circuits_from_mask(fpr_mask) -> (circuits, row_index)
        circuits   : the active circuit objects/ids for this iteration
        row_index  : (r,) int array mapping each residual row to its circuit's
                     position in `circuits` (so shots can be aggregated per circuit)
    current_shots(circuits) -> (r,) cumulative shots already spent on each row
    add_shots(circuits, extra_per_circuit) -> None
        draw `extra_per_circuit[j]` NEW shots for circuits[j] and MERGE the counts
        into the dataset that fun() reads (accumulate, do not replace)
"""

from __future__ import annotations

import numpy as np

import adaptive_shots as ashots


def make_adaptive_shot_hook(
    *,
    probability_and_jacobian,
    circuits_from_mask,
    current_shots,
    add_shots,
    schedule=None,
    criterion="D",
    metric_M=None,
    variance_floor=1e-12,
    logger=None,
    per_iter_report=None,
    ensure_baseline=None,
    total_shot_budget=None,
    accounted_shots=None,
    allocate_every=1,
    per_circuit_allocation=True,
):
    """Return a `hook(state)` callable for gradient_pounders.pouders(iter_callback=...).

    Parameters
    ----------
    probability_and_jacobian, circuits_from_mask, current_shots, add_shots
        Problem-specific callables (see module docstring).
    schedule : callable(state) -> int, optional
        Maps the POUNDERS state to this iteration's shot budget N_k.  Defaults to
        Jeff's first-pass geometric schedule N_k = 32 * 1.5**iteration (capped).
        Use `geometric_budget(...)` or `adaptive_budget(...)` below, or your own.
    criterion : {"D", "A", "L"}
        Design criterion forwarded to adaptive_shots.allocate_shots.  "L" (infidelity-
        aligned) additionally requires ``metric_M``.
    metric_M : (d, d) array or callable(state) -> (d, d), optional
        The infidelity metric M for criterion="L" (M = infidelity Hessian; see
        adaptive_shots.infidelity_metric_hessian).  Pass a fixed array to reuse one
        metric every iteration (cheap), or a callable to rebuild it from the current
        state (expensive -- the callable should cache).  Ignored for D/A.
    logger : callable(str), optional
        Optional logging sink.
    total_shot_budget : int, optional
        Global revealed-shot budget. A final oversized batch is clipped to the
        remaining budget. Once exhausted, shot acquisition stops but POUNDERS
        continues optimizing on the accumulated dataset.
    accounted_shots : callable(state) -> int, optional
        Returns cumulative revealed shots. Required with ``total_shot_budget``.
    """
    if schedule is None:
        schedule = geometric_budget()
    if total_shot_budget is not None:
        total_shot_budget = int(total_shot_budget)
        if total_shot_budget <= 0:
            raise ValueError("total_shot_budget must be positive.")
        if accounted_shots is None:
            raise ValueError("accounted_shots is required with total_shot_budget.")

    def _log(msg):
        if logger is not None:
            logger(msg)

    def hook(state):
        # 0. Allocation cadence. The D/A/L design is re-solved from scratch on every call
        #    (Frank-Wolfe up to 200 iterations, then greedy integer rounding over every
        #    circuit), which is by far the most expensive thing this hook does -- and it
        #    dominates the whole run once FPR is off and the design spans all circuits.
        #    The Fisher information barely moves between consecutive POUNDERS steps, so
        #    re-solving every step is mostly wasted. allocate_every=k solves on one call in k
        #    and skips the rest, then multiplies that call's N_k by k so the SAME total budget
        #    is still spent -- in k-times larger, k-times rarer batches. Without that
        #    multiplier the arm would silently under-spend its budget by a factor of k and
        #    stop being budget-matched against the other arms.
        if allocate_every > 1:
            _it = int(state.get("iteration", 0) or 0)
            if _it % int(allocate_every) != 0:
                return {
                    "data_changed": False,
                    "requested_budget": 0,
                    "scheduled_budget": 0,
                    "shots_added": 0,
                    "skipped_for_cadence": True,
                }

        # Optional per-iteration report (e.g. infidelity-to-truth), logged each call.
        if per_iter_report is not None:
            try:
                _rep = per_iter_report(state)
                if _rep:
                    _log(_rep)
            except Exception as exc:
                _log(f"per_iter_report failed: {exc!r}")
        # 1. How many new shots this iteration (the schedule -- "how many").
        # x allocate_every so skipping does not reduce the total spend (see the cadence note)
        requested_N_k = int(schedule(state)) * max(int(allocate_every), 1)
        N_k = requested_N_k
        budget_was_clipped = False
        previous_rho = state.get("previous_rho")
        rho_text = "unavailable" if previous_rho is None else f"{float(previous_rho):.6g}"

        spent = None
        remaining = None
        if total_shot_budget is not None:
            spent = int(accounted_shots(state))
            remaining = int(total_shot_budget - spent)
            if remaining <= 0:
                _log(
                    f"iter {state.get('iteration')}: adaptive shot budget exhausted "
                    f"(spent={spent}, total budget={total_shot_budget}); "
                    "continuing optimization without new shots."
                )
                return {
                    "data_changed": False,
                    "requested_budget": requested_N_k,
                    "scheduled_budget": 0,
                    "accounted_shots": spent,
                    "remaining_budget": 0,
                    "shot_budget_exhausted": True,
                }
            if N_k > remaining:
                _log(
                    f"iter {state.get('iteration')}: clipping requested N_k={N_k} "
                    f"to the remaining global shot budget={remaining}."
                )
                N_k = remaining
                budget_was_clipped = True

        if N_k <= 0:
            _log(
                f"iter {state.get('iteration')}: no adaptive top-up "
                f"(previous rho={rho_text})."
            )
            return {
                "data_changed": False,
                "requested_budget": requested_N_k,
                "scheduled_budget": N_k,
                "accounted_shots": spent,
                "remaining_budget": remaining,
                "shot_budget_exhausted": False,
            }

        # 2. Which circuits/rows are active at this incumbent (online FPR set).
        circuits, row_index = circuits_from_mask(state.get("fpr_mask"))
        row_index = np.asarray(row_index, dtype=int).reshape(-1)
        n_circuits = len(circuits)
        if n_circuits == 0:
            return {
                "data_changed": False,
                "requested_budget": requested_N_k,
                "scheduled_budget": N_k,
                "accounted_shots": spent,
                "remaining_budget": remaining,
                "shot_budget_exhausted": False,
            }

        # 2b. Lean baseline: experimentally measure any NEWLY-revealed circuit before
        #     allocating, so only circuits FPR selects are ever sampled.
        _piloted = int(ensure_baseline(circuits) or 0) if ensure_baseline is not None else 0

        # A baseline callback may itself acquire shots. Recheck the global budget
        # before placing the optimized top-up so the complete acquisition event,
        # not only the scheduled allocation, respects the limit.
        if total_shot_budget is not None and _piloted > 0:
            spent = int(accounted_shots(state))
            remaining = max(0, int(total_shot_budget - spent))
            if N_k > remaining:
                _log(
                    f"iter {state.get('iteration')}: baseline sampling left "
                    f"{remaining} shots; clipping optimized top-up from {N_k}."
                )
                N_k = remaining
                budget_was_clipped = True
            if N_k <= 0:
                return {
                    "data_changed": True,
                    "shots_added": 0,
                    "piloted": _piloted,
                    "requested_budget": requested_N_k,
                    "scheduled_budget": 0,
                    "accounted_shots": spent,
                    "remaining_budget": remaining,
                    "shot_budget_exhausted": remaining == 0,
                }

        # 3. Model p, J at x_k, Bernoulli single-shot variances, cumulative shots.
        #    A selection-aware GST oracle already computed these quantities for
        #    POUNDERS' center model. Reuse them when the masks agree.
        x = np.asarray(state["x"], dtype=float)
        requested_mask = state.get("fpr_mask")
        cached_mask = state.get("center_probability_mask")
        cached_p = np.asarray(state.get("center_probabilities", []), dtype=float).reshape(-1)
        cached_J = np.asarray(
            state.get("center_probability_jacobian", []), dtype=float
        )
        use_center_cache = (
            requested_mask is not None
            and cached_mask is not None
            and np.array_equal(
                np.asarray(requested_mask, dtype=bool).reshape(-1),
                np.asarray(cached_mask, dtype=bool).reshape(-1),
            )
            and cached_p.shape == (row_index.size,)
            and cached_J.ndim == 2
            and cached_J.shape[0] == row_index.size
        )
        if use_center_cache:
            p, J = cached_p, cached_J
            _log(
                f"iter {state.get('iteration')}: reused center probabilities/Jacobian "
                "for adaptive allocation."
            )
        else:
            p, J = probability_and_jacobian(x, circuits)
            p = np.asarray(p, dtype=float).reshape(-1)
            J = np.asarray(J, dtype=float)
        sigma2 = np.maximum(p * (1.0 - p), variance_floor)
        n_now = np.asarray(current_shots(circuits), dtype=float).reshape(-1)

        # 4. Shot allocation.
        #
        # PER-CIRCUIT (default) is the physically correct design: a circuit-shot samples all
        # of a circuit's outcomes at once, so the decision variable is one integer per circuit
        # and its per-shot information is the multinomial block
        # B_s = sum_beta J_{s,beta}^T J_{s,beta} / p_{s,beta}.
        #
        # PER-ROW (per_circuit_allocation=False) is the older path: it optimises over
        # (circuit, outcome) rows weighted by 1/(p(1-p)) -- i.e. it may ask for a number of
        # shots on one OUTCOME, which is not purchasable -- and then max-aggregates rows back
        # to circuits.
        #
        # For a BINARY measurement the two give identical circuit rankings, which was verified
        # numerically on this model: both outcome rows of a circuit have the same leverage
        # (J_1 = -J_0) and the same p(1-p), so max_row L/(p(1-p)) equals sum_beta L/p_beta
        # exactly (correlation 1.0000000000, Spearman 1.000000, max |ratio-1| = 2.7e-06 from
        # Jacobian finite-difference noise). The per-row path is kept only to reproduce older
        # runs; it is NOT correct for more than two outcomes, where sum_beta 1/p_beta does not
        # collapse to 1/(p(1-p)) and the rows of a circuit no longer share a leverage.
        M = metric_M(state) if callable(metric_M) else metric_M
        if per_circuit_allocation or budget_was_clipped:
            # N_k now equals the remaining physical circuit-shot budget. Use the
            # circuit-level allocator for this final batch because it guarantees
            # sum(extra_per_circuit) == N_k. Max-aggregating a row allocation can
            # otherwise leave part of the global budget unspent.
            n_circuit = np.zeros(n_circuits, dtype=float)
            np.maximum.at(n_circuit, row_index, n_now)
            extra_per_circuit, info = ashots.allocate_shots_per_circuit(
                J,
                p,
                N_k,
                row_index,
                n_circuit=n_circuit,
                criterion=criterion,
                metric_M=M,
            )
            fw_gap = float(info["gap"])
        else:
            extra_rows, info = ashots.allocate_shots(
                J, sigma2, N_k, n=n_now, criterion=criterion, metric_M=M)
            extra_per_circuit = np.zeros(n_circuits, dtype=int)
            np.maximum.at(extra_per_circuit, row_index, extra_rows.astype(int))
            fw_gap = float(info["fw"]["gap"])
        added = int(extra_per_circuit.sum())
        if added > 0:
            add_shots(circuits, extra_per_circuit)
        if added <= 0 and _piloted <= 0:
            return {
                "data_changed": False,
                "requested_budget": requested_N_k,
                "scheduled_budget": N_k,
                "accounted_shots": spent,
                "remaining_budget": remaining,
                "shot_budget_exhausted": False,
            }   # nothing measured this iteration
        _log(
            f"iter {state.get('iteration')}: N_k={N_k}, placed {added} shots over "
            f"{int(np.count_nonzero(extra_per_circuit))}/{n_circuits} circuits "
            f"(previous rho={rho_text}; piloted {_piloted} new; criterion={criterion}, "
            f"FW gap={fw_gap:.2e})."
        )
        post_spent = int(accounted_shots(state)) if total_shot_budget is not None else None
        post_remaining = (
            max(0, int(total_shot_budget - post_spent))
            if total_shot_budget is not None else None
        )
        return {
            "data_changed": True,
            "shots_added": added,
            "piloted": _piloted,
            "extra_per_circuit": extra_per_circuit,
            "requested_budget": requested_N_k,
            "scheduled_budget": N_k,
            "accounted_shots": post_spent,
            "remaining_budget": post_remaining,
            "shot_budget_exhausted": post_remaining == 0 if post_remaining is not None else False,
        }

    return hook


# ---------------------------------------------------------------------------
# Ready-made schedules (thin wrappers over adaptive_shots, matched to POUNDERS state)
# ---------------------------------------------------------------------------


def geometric_budget(base=1.5, n0=32, n_min=None, n_max=8192):
    """Jeff's first pass: N_k = n0 * base**k (base<2), capped. State-only."""
    def schedule(state):
        return ashots.geometric_schedule(
            int(state.get("iteration", 0)), base=base, n0=n0, n_min=n_min, n_max=n_max
        )
    return schedule


def rho_gated_geometric_budget(
    base=1.5,
    n0=32,
    n_min=None,
    n_max=8192,
    rho_band=0.05,
    initial_budget=0,
):
    """Spend a geometric shot batch only after a borderline previous rho.

    The callback at iteration k receives rho from iteration k-1. A shot batch
    is requested when ``abs(previous_rho) < rho_band``. Skipped optimization
    iterations do not advance the geometric schedule, so the j-th actual shot
    update receives ``n0 * base**j`` rather than a budget based on the POUNDERS
    iteration number.
    """
    if rho_band <= 0:
        raise ValueError("rho_band must be positive.")
    if initial_budget < 0:
        raise ValueError("initial_budget must be nonnegative.")

    shot_update_index = 0

    def schedule(state):
        nonlocal shot_update_index

        previous_rho = state.get("previous_rho")
        if previous_rho is None:
            return int(initial_budget)

        previous_rho = float(previous_rho)
        if not np.isfinite(previous_rho) or abs(previous_rho) >= float(rho_band):
            return 0

        budget = ashots.geometric_schedule(
            shot_update_index,
            base=base,
            n0=n0,
            n_min=n_min,
            n_max=n_max,
        )
        shot_update_index += 1
        return int(budget)

    return schedule


def rho_gated_delta_inverse_square_budget(
    constant=0.03,
    n_min=0,
    delta_floor=1e-12,
    rho_band=0.05,
    initial_budget=0,
):
    """Spend a ``constant / Delta_k**2`` batch after borderline rho.

    The callback at iteration k receives the current outer POUNDERS trust-region
    radius ``Delta_k`` and rho from iteration k-1. A top-up is requested only
    when ``abs(previous_rho) < rho_band``. No upper cap is applied here; use
    ``total_shot_budget`` in ``make_adaptive_shot_hook`` as the stopping rule.
    """
    if constant <= 0:
        raise ValueError("constant must be positive.")
    if n_min < 0:
        raise ValueError("n_min must be nonnegative.")
    if delta_floor <= 0:
        raise ValueError("delta_floor must be positive.")
    if rho_band <= 0:
        raise ValueError("rho_band must be positive.")
    if initial_budget < 0:
        raise ValueError("initial_budget must be nonnegative.")

    def schedule(state):
        previous_rho = state.get("previous_rho")
        if previous_rho is None:
            return int(initial_budget)

        previous_rho = float(previous_rho)
        if not np.isfinite(previous_rho) or abs(previous_rho) >= float(rho_band):
            return 0

        delta = float(state.get("delta", np.nan))
        if not np.isfinite(delta):
            return int(n_min)

        effective_delta = max(abs(delta), float(delta_floor))
        budget = max(int(n_min), int(np.ceil(float(constant) / effective_delta**2)))
        return int(budget)

    return schedule


def lazy_delta_inverse_square_budget(
    base=1.5,
    n0=32,
    constant=0.03,
    n_min=0,
    delta_floor=1e-12,
    initial_budget=0,
):
    """Grow batches lazily while respecting the ``Delta_k^-2`` target.

    After the optional iteration-zero top-up, iteration ``k`` requests

    ``N_k = min(ceil(n0 * base**(k-1)), ceil(constant / Delta_k**2))``.

    The geometric term prevents an abrupt jump to a huge inverse-square batch,
    while the second term ties sampling precision to the current trust-region
    radius. The global limit is enforced separately by
    :func:`make_adaptive_shot_hook`.
    """
    if not 1.0 < float(base) < 2.0:
        raise ValueError("base must be strictly between 1 and 2.")
    if n0 <= 0:
        raise ValueError("n0 must be positive.")
    if constant <= 0:
        raise ValueError("constant must be positive.")
    if n_min < 0:
        raise ValueError("n_min must be nonnegative.")
    if delta_floor <= 0:
        raise ValueError("delta_floor must be positive.")
    if initial_budget < 0:
        raise ValueError("initial_budget must be nonnegative.")

    def schedule(state):
        iteration = int(state.get("iteration", 0))
        if iteration <= 0:
            return int(initial_budget)

        delta = float(state.get("delta", np.nan))
        if not np.isfinite(delta):
            return int(n_min)

        effective_delta = max(abs(delta), float(delta_floor))
        geometric = int(np.ceil(float(n0) * float(base) ** (iteration - 1)))
        inverse_square = int(np.ceil(float(constant) / effective_delta**2))
        return int(max(int(n_min), min(geometric, inverse_square)))

    return schedule


def lazy_delta_inverse_fourth_budget(
    base=1.5,
    n0=32,
    constant=3e-4,
    n_min=0,
    delta_floor=1e-12,
    initial_budget=0,
):
    """Grow batches lazily while respecting a ``Delta_k^-4`` target.

    After the optional iteration-zero top-up, iteration ``k`` requests

    ``N_k = min(ceil(n0 * base**(k-1)), ceil(constant / Delta_k**4))``.

    The geometric term prevents an abrupt jump to a huge fourth-power batch.
    The global shot budget is enforced separately by
    :func:`make_adaptive_shot_hook`.
    """
    if not 1.0 < float(base) < 2.0:
        raise ValueError("base must be strictly between 1 and 2.")
    if n0 <= 0:
        raise ValueError("n0 must be positive.")
    if constant <= 0:
        raise ValueError("constant must be positive.")
    if n_min < 0:
        raise ValueError("n_min must be nonnegative.")
    if delta_floor <= 0:
        raise ValueError("delta_floor must be positive.")
    if initial_budget < 0:
        raise ValueError("initial_budget must be nonnegative.")

    def schedule(state):
        iteration = int(state.get("iteration", 0))
        if iteration <= 0:
            return int(initial_budget)

        delta = float(state.get("delta", np.nan))
        if not np.isfinite(delta):
            return int(n_min)

        effective_delta = max(abs(delta), float(delta_floor))
        geometric = int(np.ceil(float(n0) * float(base) ** (iteration - 1)))
        inverse_fourth = int(np.ceil(float(constant) / effective_delta**4))
        return int(max(int(n_min), min(geometric, inverse_fourth)))

    return schedule


def adaptive_budget(scale=1.0, n_min=32, n_max=8192, use="ng"):
    """Matt's adaptive: N_k ~ 1/||g_k|| (use='ng'), capped.

    Falls back to n_min when the gradient norm is unavailable (nan on the first
    iteration) so the run always makes progress.
    """
    def schedule(state):
        g = state.get("ng", np.nan)
        if not np.isfinite(g) or g <= 0:
            return int(n_min)
        return ashots.adaptive_schedule(
            grad_norm=float(g), scale=scale, n_min=n_min, n_max=n_max
        )
    return schedule
