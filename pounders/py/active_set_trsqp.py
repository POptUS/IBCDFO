"""
================================================================================
 Active-Set TR-SQP for Derivative-Free Optimization with Inequality Constraints
================================================================================

Implementation of Algorithm 1, "Active-Set TR-SQP DFO Algorithm with Criticality
Measure" (LaTeX label `alg.main_alg_chi`), for

        min_{x}  f(x)      s.t.   c_i(x) <= 0,  i = 1..p        (opt_prob)

where f and every c_i are black boxes: evaluations only, no derivatives, and each
evaluation is expensive.  All iterates stay FEASIBLE.

--------------------------------------------------------------------------------
HOW TO READ THIS FILE ALONGSIDE THE WRITE-UP
--------------------------------------------------------------------------------
Every block below is tagged with the label it implements, e.g.

        # === [step.sk] ==========
        # (LaTeX: \\nllabel{step.sk}, subproblem \\eqref{eq:step_subproblem})

so the code can be diffed against the pseudocode line by line.  Tags used:

    line 4                      build/update the linear models m^f, m^{c_i}
    [eq:active_subproblem]      A_k <- { i : max_{s in B(0,Delta_k)} m^{c_i} >= 0 }
    [step.criticality]          while Delta_k > mu*chi_k(x_k,A_k): Delta_k *= gamma_dec
    [step.mainwhile]            while true
    [step.sk]                   choose H_k > 0, solve for (s_k, tau_k)
    [line:eval_ci]              evaluate c_i(x_k+s_k); first violation sets i_viol
    [line:iviol_zero]           i_viol = 0 and s_k != 0  ->  rho_k test
    [line:very_successful]        rho_k >= eta_2   accept, Delta *= gamma_inc
    [line:successful]             rho_k >= eta_1   accept, Delta unchanged
    [line:unsuccessful]           else             reject, Delta *= gamma_dec
    [line:expand_active_set]    i_viol >= 1, i_viol not in A_k, s_k != 0
    [step.critical_verify]        Delta_k <= mu*chi_k(x_k, Atilde_k) ?
    [step.critical_verify_update]   yes -> A_k <- Atilde_k, re-solve
                                    no  -> reject, Delta *= gamma_dec, break
    [line:rejected_viol]        otherwise: reject, Delta *= gamma_dec, break

Anything the pseudocode does NOT specify is marked  [ADDED]  and justified where
it appears.  There are six such additions; they are listed at the end of this
docstring so a reviewer can find them all quickly.

--------------------------------------------------------------------------------
THE TWO MEASURES
--------------------------------------------------------------------------------
Active set (eq:active_subproblem).  Because the models are LINEAR and the trust
region is the inf-norm ball, the max in the definition is available in closed
form: max_{||s||_inf <= Delta} a^T s = Delta*||a||_1.  Hence

        A_k = { i : c_i(x_k) + Delta_k * ||g_k^{c_i}||_1 >= 0 }

i.e. "the linear model of c_i can reach 0 somewhere inside the trust region."
Only constraints in A_k are imposed in the subproblem.

Criticality measure (Section 4).  With true gradients,

        chi(x,A) = -min_{||s|| <= 1} max{ grad f(x)^T s, max_{i in A} grad c_i(x)^T s }

and chi_k is the same expression with the MODEL gradients g_k^f, g_k^{c_i}.  It is
>= 0, and (Lemma lem.critical_point) vanishes at a feasible x exactly when x is a
Fritz-John point.  Lemma lem.chis gives |chi_k - chi| <= kappa_chi * Delta_k, so
chi_k is only trustworthy when Delta_k is small -- which is why the convergence
test below is a JOINT test on chi_k and Delta_k, never on chi_k alone.

chi_k is computed by reusing the subproblem solver with H = 0, Delta = 1, and
c_active = 0 (gradients only); then chi_k = -tau.  See compute_chi().

--------------------------------------------------------------------------------
THE [ADDED] ITEMS  (not in the pseudocode)
--------------------------------------------------------------------------------
 A1  Termination.  The pseudocode is an infinite `for k`; a program needs exits.
     Four are implemented; only ONE certifies stationarity.  See "TERMINATION".
 A2  Delta floor / evaluation budget inside the criticality loop, so a run that
     is not converging stops instead of spinning.
 A3  Variable bounds l <= x <= u intersected into the step box and the chi box.
     The pseudocode writes only B(0,.); real (CUTEst) problems have bounds.
 A4  H_k = I.  The pseudocode says "choose H_k > 0" without specifying which.
 A5  Violation test uses c_i(x_k+s_k) > 1e-9 rather than > 0, so that a point
     sitting exactly on a constraint boundary is not called infeasible by
     floating-point noise.
 A6  delta_max, an optional cap on Delta growth (pseudocode has none).

--------------------------------------------------------------------------------
KNOWN COST CHARACTERISTIC (worth discussing)
--------------------------------------------------------------------------------
Line 4 of the algorithm rebuilds models for ALL p constraints, and the criticality
loop re-executes line 4 after every shrink.  So the active set currently reduces
the SIZE OF THE SUBPROBLEM but not the NUMBER OF CONSTRAINT EVALUATIONS -- on the
2-D demo the split is roughly p f-evals per c-eval.  Reducing constraint
evaluations was the original motivation, so this is the natural place for a
constraint-aware model-building routine to replace formquad.
"""

import numpy as np
from .formquad import formquad
from .sqp_subproblem import solve_subproblem
from .bmpts import bmpts

# formquad poisedness parameters:
#   Pars[0]  tight-ball radius multiple -- decides the returned `valid` flag,
#            i.e. whether the model may be treated as fully linear
#   Pars[1]  loose-ball radius multiple -- points still usable for the fit
#   Pars[2], Pars[3]  pivot thresholds;   Pars[4]  point-ordering flag
Pars = [np.sqrt(2), max(10, np.sqrt(2)), 1e-3, 1e-3, 0]


def run(f, constraints, x0, delta0=1.0, nf_max=20000, delta_max=np.inf,
        underline_delta=1e-3, mu=0.1, chi_tol=1e-3,
        delta_term=1e-4, delta_min=1e-5,
        low=None, upp=None, eta1=0.1, eta2=0.7, gamma_dec=0.5, gamma_inc=2.0,
        verbose=True):
    """
    Parameters
    ----------
    FROM THE ALGORITHM'S `Require` LIST
      f, constraints  black-box callables x -> float; feasibility is c_i(x) <= 0
      x0              starting point; the Require list demands c(x0) <= 0
      delta0          Delta_0, required to lie in [underline_delta, inf)
      underline_delta \\underline\\Delta in (0,inf).  Post-acceptance RESET on the
                      radius: every accepted step leaves Delta_{k+1} >= this value.
                      Not typical of DFO methods -- it is what supplies the uniform
                      lower bound Delta_acc used near a non-Fritz-John limit point
                      in the proof of the main theorem.
      mu              mu in (0, 1/(zeta_bar * n)); sets the criticality threshold
                      Delta_k <= mu*chi_k that must hold before a step is computed
      gamma_dec       in (0,1), shrink factor
      gamma_inc       in [1,inf), growth factor
      eta1            in (0,1),      accept threshold
      eta2            in [eta1,1),   "very successful" threshold

    [ADDED] -- NOT in the pseudocode
      chi_tol         epsilon_chi: stationarity bar for the convergence test
      delta_term      epsilon_Delta: resolution at which a small chi_k is believed.
                      Needed because Lemma lem.chis only bounds |chi_k - chi| by
                      kappa_chi*Delta_k, so chi_k alone certifies nothing.
      delta_min       give-up floor.  Delta below this => stop WITHOUT certifying.
                      Keep delta_min well below delta_term, otherwise a run can
                      hit the floor while still inside the certification window.
      nf_max          evaluation budget (f-evals + c-evals)
      delta_max       optional cap on Delta growth; default inf = uncapped
      low, upp        variable bounds; default to a large box
      verbose         print the per-iteration trace
    """
    p = len(constraints)
    x0 = np.asarray(x0, float); n = len(x0)
    if low is None: low = -1e3 * np.ones(n)          # [ADDED A3]
    if upp is None: upp = 1e3 * np.ones(n)
    np_max = 2 * n + 1                               # max points in a model
    delta0 = max(delta0, underline_delta)            # Require: Delta_0 >= udelta

    # Require: x_0 with c(x_0) <= 0.  The method is FEASIBLE -- it never leaves
    # the feasible region, so it cannot be started outside it.
    assert all(constraints[i](x0) <= 0 for i in range(p)), "x0 must satisfy c(x0) <= 0"

    # ======================================================================
    # EVALUATION CACHE
    # ======================================================================
    # Every point ever evaluated is stored once, with its f value and its p
    # constraint values (NaN = not yet evaluated).  Two reasons this matters:
    #   * f and each c_i are evaluated INDEPENDENTLY -- the problem setting
    #     allows evaluating any subset of {f, c_1..c_p} at a given x, which is
    #     exactly what the active-set idea needs.
    #   * a point is never re-evaluated, so the evaluation counters below are a
    #     true count of black-box calls, which is the cost measure that matters.
    pts, fval, cval = [], [], [[] for _ in range(p)]
    evf, evc = [0], [0]                              # [f-evals], [c-evals]

    def pidx(x):
        """Index of x in the cache, appending a fresh NaN row if it is new."""
        x = np.asarray(x, float)
        for i, q in enumerate(pts):
            if np.allclose(q, x, atol=1e-12):
                return i
        pts.append(x); fval.append(np.nan)
        for i in range(p): cval[i].append(np.nan)
        return len(pts) - 1

    def eval_f(x):
        """f(x), evaluated at most once per distinct x."""
        i = pidx(x)
        if np.isnan(fval[i]): fval[i] = f(x); evf[0] += 1
        return fval[i]

    def eval_c(x, j):
        """c_j(x), evaluated at most once per distinct (x, j)."""
        i = pidx(x)
        if np.isnan(cval[j][i]): cval[j][i] = constraints[j](x); evc[0] += 1
        return cval[j][i]

    # ======================================================================
    # MODEL BUILDING  (formquad)
    # ======================================================================
    def build_model(values, eval_one, x_k, delta, improve=True, max_rounds=3):
        """Fit a LINEAR model of one black box around x_k on B(x_k, delta).

        Returns (G, H, valid) where G is the model gradient and `valid` is
        formquad's poisedness flag.  `valid` is exactly the hypothesis of
        Assumption assm:interpolation (fully linear on the trust region) -- if it
        is False, the Taylor-like error bounds, and therefore Lemma lem.chis, do
        not apply.  It is recorded and reported but never used to alter a
        decision, so the algorithm's control flow matches the pseudocode.

        If the geometry is not poised, formquad returns directions Mdir; bmpts
        makes those directions respect the bounds, and we evaluate along them to
        improve the geometry, then refit (at most max_rounds times).
        """
        for _ in range(max_rounds):
            v = np.array(values)
            mask = ~np.isnan(v)                      # only points where this
            Xs = np.array(pts)[mask]                 # black box has been evaluated
            Vs = v[mask].reshape(-1, 1)

            xk_idx = int(np.argmin(np.linalg.norm(Xs - x_k, axis=1)))

            Mdir, mp, valid, G, H, Mind = formquad(Xs,Vs,delta,xk_idx,np_max,Pars,False,)

            G_arr = np.asarray(G)

            # Poised, or improvement not requested, or nothing to improve with.
            if not improve or valid or np.asarray(Mdir).size == 0:
                break

            theta = Pars[2]

            try:
                # bmpts projects the model-improving directions into the bounds.
                Mdir1, mp1 = bmpts(x_k, np.atleast_2d(Mdir)[0:n - mp, :],
                                   low, upp, delta, theta)
                Mdir1 = np.atleast_2d(Mdir1)

                for i in range(n - mp1):
                    D = Mdir1[i, :]
                    eval_one(np.clip(x_k + D, low, upp))

            except Exception as e:
                # Fall back to the raw formquad directions, scaled by delta.
                for d in np.atleast_2d(Mdir):
                    eval_one(np.clip(x_k + delta * d, low, upp))

        G = np.asarray(G).reshape(-1)

        H = (
            np.asarray(H).reshape(n, n)
            if np.asarray(H).size
            else np.zeros((n, n))
        )

        return G, H, bool(valid)

    # ----------------------------------------------------------------------
    # line 4  +  [eq:active_subproblem]
    # ----------------------------------------------------------------------
    def build_all(x_k, delta):
        """Line 4 and eq:active_subproblem together.

        Line 4:   build/update the linear models
                      m^f(x_k+s)     = f(x_k)   + (g_k^f)^T s
                      m^{c_i}(x_k+s) = c_i(x_k) + (g_k^{c_i})^T s     for ALL i
        eq:active_subproblem:
                      A_k = { i : max_{s in B(0,Delta_k)} m^{c_i}(x_k+s) >= 0 }
                          = { i : c_i(x_k) + Delta_k*||g_k^{c_i}||_1 >= 0 }
        the second form because the models are linear and the ball is inf-norm.

        Returns None if the objective model cannot be built (formquad returned an
        empty gradient) -- handled by the caller as a non-certifying exit.

        NOTE: models are built for all p constraints, per line 4 as written.  The
        active set narrows the SUBPROBLEM, not the evaluation count.
        """
        g_f, _H, valid_f = build_model(fval, eval_f, x_k, delta, improve=True)
        if g_f.size != n:
            return None
        cmods, cvalid = {}, {}
        for j in range(p):
            # the double lambda binds j by value (not by closure) for each model
            gj, _, vj = build_model(cval[j],
                                    (lambda jj: (lambda z: eval_c(z, jj)))(j),
                                    x_k, delta, improve=True)
            cmods[f"c{j+1}"] = (eval_c(x_k, j), gj)   # (c_i(x_k), g_k^{c_i})
            cvalid[f"c{j+1}"] = vj
        A_k = [nm for nm, (val, g) in cmods.items()
               if val + delta * np.sum(np.abs(g)) >= 0]     # >= 0, as written
        return g_f, valid_f, cmods, cvalid, A_k

    # ----------------------------------------------------------------------
    # CRITICALITY MEASURE  chi_k
    # ----------------------------------------------------------------------
    def compute_chi(x_k, g_f, cmods, A_k):
        """chi_k(x_k, A) = -min_{s in B(0,1)} max{ (g^f)^T s, max_{i in A} (g^{c_i})^T s }

        Obtained from the same solver as the step subproblem, with:
            H = 0          -> the objective is just tau, so the QP becomes the LP
                              min tau  s.t.  (g^f)^T s <= tau,  (g^{c_i})^T s <= tau
            Delta = 1      -> the unit ball in the definition
            c_active = 0   -> chi uses GRADIENTS ONLY; the constants c_i(x_k) do
                              NOT appear (this differs from the step subproblem,
                              which does carry them)
        then chi_k = -tau.  The max(0, .) guards against tiny negative values from
        the solver; chi >= 0 always, since s = 0 is feasible and gives 0.

        [ADDED A3] the box low <= x_k + s <= upp is intersected with the unit ball.
        """
        if A_k:
            G_active = np.array([cmods[nm][1] for nm in A_k])
        else:
            G_active = np.zeros((0, n))
        c_active = np.zeros(len(A_k))                 # gradients only
        _, tau, _ = solve_subproblem(x_k, g_f, np.zeros((n, n)),
                                     c_active, G_active, 1.0, low, upp)
        return max(0.0, -float(tau))

    # ======================================================================
    # INITIALIZE
    # ======================================================================
    eval_f(x0)
    for j in range(p): eval_c(x0, j)
    x_k = x0.copy(); delta = delta0; k = 0
    growth = 0; terminated = False; term_reason = None
    trace_pts, trace_active, overshoots = [], [], []
    chi_k = np.inf

    # Observe-only instrumentation.  `mdl_valid` records whether every model in
    # use was fully linear; it is REPORTED but never allowed to change a branch,
    # so the executed control flow is exactly the pseudocode's.
    valid_log = []
    converged_valid = None
    a_w = max(10, 3 * p + 2)

    # ----------------------------------------------------------------------
    # TRACE: one line per outer iteration k.
    #   crit  number of times [step.criticality] halved Delta this iteration
    #   pass  number of times [step.sk] was solved (>1 means A_k grew mid-iteration)
    #   x, delta, chi, A_k are the values the STEP was actually computed with
    # ----------------------------------------------------------------------
    hdr = (f"{'k':>4} {'crit':>4} {'pass':>4} {'x[0]':>9} {'x[1]':>9} "
           f"{'delta':>9} {'|s|':>9} {'chi':>9} {'f':>12} {'A_k':>{a_w}} "
           f"{'rho':>7} {'m':>2}  branch")

    if verbose:
        print(hdr); print("-" * len(hdr))

    def row(k_, crit, npass, x_, d, sN, chi, fv, Aset, rho, mv, br):
        s_s = f"{sN:>9.2e}" if sN is not None else f"{'-':>9}"
        rho_s = f"{rho:>7.3f}" if (rho is not None and np.isfinite(rho)) else f"{'-':>7}"
        print(f"{k_:>4} {crit:>4} {npass:>4} {x_[0]:>9.4f} {x_[1]:>9.4f} "
              f"{d:>9.2e} {s_s} {chi:>9.2e} {fv:>12.5e} {Aset:>{a_w}} {rho_s} "
              f"{mv:>2}  {br}")

    # ======================================================================
    #                      OUTER LOOP   --   `for k = 0,1,2,...`
    # ======================================================================
    # The pseudocode's loop is infinite; the guards here are [ADDED A1/A2].
    while evf[0] + evc[0] < nf_max and not terminated:
        if verbose and k > 0 and k % 15 == 0:
            print(hdr); print("-" * len(hdr))

        # === line 4  +  [eq:active_subproblem] ============================
        built = build_all(x_k, delta)
        if built is None:
            # [ADDED A1, exit 4] formquad could not produce a model gradient.
            # A tooling failure, not a mathematical conclusion -- no certificate.
            terminated = True
            term_reason = f"model unbuildable at delta={delta:.2e} (NOT certified)"
            break
        g_f, valid_f, cmods, cvalid, A_k = built
        chi_k = compute_chi(x_k, g_f, cmods, A_k)
        trace_pts.append(x_k.copy())

        # === [step.criticality] ===========================================
        #     while Delta_k > mu * chi_k(x_k, A_k):
        #         Delta_k <- gamma_dec * Delta_k
        #         recompute m^f, m^{c_i}, A_k as in line 4
        #
        # Purpose: enter [step.mainwhile] only when Delta_k <= mu*chi_k.  That is
        # the hypothesis Lemma lem.cauchy needs to conclude
        #       tau_k <= -(1/2) * chi_k(x_k,A_k) * Delta_k,
        # i.e. the predicted decrease is proportional to chi_k * Delta_k.
        #
        # Lemma lem.finite: if this loop never terminates, x_k is already a
        # Fritz-John point.  So non-termination here is not a failure -- it means
        # we have arrived.  The guards below turn that into a finite exit.
        #
        # ORDERING MATTERS.  The certificate and floor checks come BEFORE the
        # rebuild.  When chi_k is tiny, mu*chi_k is tiny too, so the loop drives
        # Delta toward ~mu*chi_k and would rebuild at a radius so small that
        # formquad returns an empty gradient -- crashing one step AFTER a valid
        # certificate was already in hand.
        crit = 0
        delta_in = delta                       # Delta at the START of iteration k
        ev_in = evf[0] + evc[0]
        while delta > mu * chi_k:
            delta *= gamma_dec
            crit += 1

            # certificate already reached during the shrink -> stop, do NOT rebuild
            if chi_k <= chi_tol and delta <= delta_term:
                break
            # [ADDED A2] floor / budget -> leave the loop; handled as exits below
            if delta < delta_min or evf[0] + evc[0] >= nf_max:
                break

            g_f, valid_f, cmods, cvalid, A_k = build_all(x_k, delta)   # line 4 again
            chi_k = compute_chi(x_k, g_f, cmods, A_k)

        # === [ADDED A1, exit 1]  STATIONARITY CERTIFICATE =================
        # The ONLY exit that claims a Fritz-John point.  The test is JOINT and
        # must stay joint: chi_k is built from model gradients, and Lemma
        # lem.chis only gives |chi_k - chi| <= kappa_chi*Delta_k.  A small chi_k
        # at a large Delta_k therefore proves nothing about the true chi.
        if chi_k <= chi_tol and delta <= delta_term:
            # Was every model actually fully linear here?  If not, even the
            # lem.chis bound does not apply and the certificate is only nominal.
            converged_valid = bool(valid_f and all(cvalid.get(nm, False) for nm in A_k))
            mv = "V" if converged_valid else "."
            valid_log.append(dict(k=k, branch="converged", mdl_valid=converged_valid,
                                  delta=delta, chi=chi_k, A_k=list(A_k)))
            if verbose:
                row(k, crit, 0, x_k, delta, None, chi_k, eval_f(x_k),
                    "{"+",".join(A_k)+"}", None, mv, "CONVERGED")
            terminated = True
            term_reason = (f"converged: chi={chi_k:.2e}<=chi_tol={chi_tol:.0e} and "
                           f"delta={delta:.2e}<=delta_term={delta_term:.0e}")
            break

        # === [ADDED A1, exit 2]  TRUST-REGION COLLAPSE ====================
        # Delta fell below the give-up floor without the certificate ever firing.
        # NOT convergence: chi_k never became small at a trustworthy resolution.
        # Reported separately so it is never mistaken for success.
        if delta < delta_min:
            converged_valid = None             # explicitly claiming nothing
            valid_log.append(dict(k=k, branch="delta_floor", mdl_valid=False,
                                  delta=delta, chi=chi_k, A_k=list(A_k)))
            if verbose:
                row(k, crit, 0, x_k, delta, None, chi_k, eval_f(x_k),
                    "{"+",".join(A_k)+"}", None, ".",
                    "STOP: delta<delta_min (no certificate)")
            terminated = True
            term_reason = (f"delta floor: delta={delta:.2e}<delta_min={delta_min:.0e} "
                           f"with chi={chi_k:.2e} (NOT certified stationary)")
            break

        # === [ADDED A2]  budget guard =====================================
        # The criticality loop can also exit on the nf_max guard, in which case
        # Delta_k > mu*chi_k may STILL hold -- a state the algorithm never allows
        # at [step.mainwhile].  Stop rather than take a step whose Cauchy-decrease
        # hypothesis (Lemma lem.cauchy) is not satisfied.
        if evf[0] + evc[0] >= nf_max:
            break

        # === [step.mainwhile]   `while true` ==============================
        # Terminates (Lemma lem.well-defined) because the only way to repeat is
        # to ADD an index to A_k, no index is ever removed, and there are p of
        # them -- so at most p passes.
        inner = 0; rho = np.nan; _grew = []; _out = None
        while True:
            inner += 1
            # Only the constraints currently in A_k are imposed.  This is where
            # the active-set idea pays off: the subproblem carries |A_k| rows,
            # not p.
            if A_k:
                c_active = np.array([cmods[nm][0] for nm in A_k])
                G_active = np.array([cmods[nm][1] for nm in A_k])
            else:
                c_active = np.zeros(0); G_active = np.zeros((0, n))

            # --- [step.sk] / eq:step_subproblem ---------------------------
            #   (s_k, tau_k) in argmin_{s in B(0,Delta_k), tau}
            #                     tau + (1/2) s^T H_k s
            #        subject to   m^f(x_k+s)     <= f(x_k) + tau
            #                     m^{c_i}(x_k+s) <= tau        for all i in A_k
            #
            # The tau variable turns the min-max into one smooth QP (the
            # Topkis-Veinott reformulation).  [ADDED A4] H_k = I: the algorithm
            # only requires H_k > 0.  Positive definiteness is what makes the
            # solution unique (Lemma lem.subproblems) and the optimal value
            # strictly negative whenever s_k != 0.
            s, tau, _ = solve_subproblem(x_k, g_f, np.eye(n),
                                         c_active, G_active, delta, low, upp)
            s_norm = float(np.max(np.abs(s)))     # inf-norm: the "s_k != 0" test
            # m^f(x_k) - m^f(x_k+s_k) = -(g_k^f)^T s_k, since m^f is LINEAR.
            # Computed here rather than taken from the solver so the rho below
            # uses the model decrease the algorithm specifies, not the QP's own
            # objective (which also contains the (1/2) s^T H s regularizer).
            pred = -float(np.asarray(g_f, float) @ s)

            mdl_valid = bool(valid_f and all(cvalid.get(nm, False) for nm in A_k))
            mv = "V" if mdl_valid else "."

            d_before = delta                   # Delta before this pass changes it
            ev_m = evf[0] + evc[0]
            x_k_disp = x_k.copy()              # x BEFORE a possible accept
            Aset = "{" + ",".join(A_k) + "}"

            # --- [line:eval_ci] / [line:set_iviol] ------------------------
            # Evaluate the constraints at the trial point and stop at the FIRST
            # violation.  Stopping early is deliberate: constraint evaluations
            # are expensive, and one violation is enough to decide the branch.
            # [ADDED A5] tolerance 1e-9 instead of a bare > 0, so a point sitting
            # exactly on a boundary is not declared infeasible by rounding.
            x_trial = x_k + s
            i_viol = 0
            for i in range(p):
                if eval_c(x_trial, i) > 1e-9:
                    i_viol = i + 1; break

            # ==============================================================
            # BRANCH 1 -- [line:iviol_zero]   i_viol = 0 and s_k != 0
            # Trial point is FEASIBLE for every constraint, so judge it by rho.
            # ==============================================================
            if i_viol == 0 and s_norm > 0:
                ft = eval_f(x_trial)
                #   rho_k = ( f(x_k) - f(x_k+s_k) ) / ( m^f(x_k) - m^f(x_k+s_k) )
                rho = (eval_f(x_k) - ft) / pred if pred > 1e-14 else -np.inf
                if rho >= eta2:
                    # [line:very_successful]
                    #   x_{k+1} = x_k + s_k,  Delta_{k+1} = max{udelta, ginc*Delta_k}
                    x_k = x_trial
                    delta = max(underline_delta, min(gamma_inc * delta, delta_max))
                    br = "accept"
                elif rho >= eta1:
                    # [line:successful]
                    #   x_{k+1} = x_k + s_k,  Delta_{k+1} = max{udelta, Delta_k}
                    x_k = x_trial
                    delta = max(underline_delta, delta)
                    br = "accept"
                else:
                    # [line:unsuccessful]
                    #   x_{k+1} = x_k,  Delta_{k+1} = gamma_dec * Delta_k
                    delta *= gamma_dec
                    br = "reject: rho low"
                valid_log.append(dict(k=k, inner=inner, branch=br, mdl_valid=mdl_valid,
                                      delta=delta, chi=chi_k, rho=rho))
                _out = (x_k_disp, d_before, s_norm, ft, Aset, rho, mv, br)
                break

            # ==============================================================
            # BRANCH 2 -- [line:expand_active_set]
            #   i_viol >= 1,  i_viol NOT in A_k,  s_k != 0
            # A constraint we were not modelling has blocked the step.  This is
            # the case the active-set estimate got wrong: the linear model said
            # c_{i_viol} was safe across the whole trust region, but the true
            # (curved) constraint is violated at the trial point.
            # ==============================================================
            elif i_viol >= 1 and (f"c{i_viol}" not in A_k) and s_norm > 0:
                jj = i_viol - 1
                overshoots.append((x_k.copy(), x_trial.copy()))   # instrumentation

                #   Atilde_k <- A_k U {i_viol};  build/update m^{c_i}, i = i_viol
                #
                # Built into SHADOW copies on purpose.  The algorithm commits the
                # augmented set only at [step.critical_verify_update], i.e. only
                # if the test below passes; writing straight into A_k/cmods here
                # would commit it unconditionally.
                gj, _, vj = build_model(cval[jj], (lambda z: eval_c(z, jj)),
                                        x_k, delta, improve=True)
                A_k_tilde = A_k + [f"c{i_viol}"]
                cmods_tilde = dict(cmods)
                cmods_tilde[f"c{i_viol}"] = (eval_c(x_k, jj), gj)

                # --- [step.critical_verify] -------------------------------
                #        is  Delta_k <= mu * chi_k(x_k, Atilde_k) ?
                #
                # Adding a constraint can only DECREASE chi_k (Lemma
                # lem.ph_augment: chi_k(x,A) >= chi_k(x,Abar) for A subset Abar),
                # so the criticality condition that held for A_k may FAIL for
                # Atilde_k.  This test re-establishes it.
                #
                # Why it matters: the proof of Lemma lem.cauchy argues that if
                # [step.sk] is reached again, it is only after this test has
                # verified Delta_k <= mu*chi_k for the CURRENT active set.  Skip
                # it and the Cauchy-decrease bound no longer covers the re-solved
                # subproblem.
                chi_tilde = compute_chi(x_k, g_f, cmods_tilde, A_k_tilde)

                if delta <= mu * chi_tilde:
                    # --- [step.critical_verify_update]  A_k <- Atilde_k ----
                    # Commit, then fall through to re-solve [step.sk] with the
                    # larger active set.  This is the only way `pass` exceeds 1.
                    cmods = cmods_tilde
                    cvalid[f"c{i_viol}"] = vj
                    A_k = A_k_tilde
                    chi_k = chi_tilde
                    growth += 1
                    valid_log.append(dict(k=k, inner=inner, branch=f"grow c{i_viol}",
                                          mdl_valid=mdl_valid, delta=delta, chi=chi_k))
                    _grew.append(f"c{i_viol}")
                else:
                    # --- else-branch of [step.critical_verify] -------------
                    #   x_{k+1} <- x_k,  Delta_{k+1} <- gamma_dec*Delta_k, break
                    # The augmented set would violate the criticality condition,
                    # so the discovery is made at a radius too large to trust:
                    # discard it and shrink instead.
                    delta *= gamma_dec
                    valid_log.append(dict(k=k, inner=inner,
                                          branch="crit_verify_fail,shrink",
                                          mdl_valid=mdl_valid, delta=delta,
                                          chi=chi_tilde))
                    _out = (x_k.copy(), d_before, s_norm, eval_f(x_k), Aset,
                            None, mv, f"reject: c{i_viol} verify failed")
                    break

            # ==============================================================
            # BRANCH 3 -- [line:rejected_viol]  (the algorithm's final Else)
            # Either a constraint ALREADY in A_k was violated (the model was
            # imposed but was not accurate enough at this radius), or s_k = 0.
            # Nothing to learn by growing A_k, so shrink and move on.
            #   x_{k+1} = x_k,  Delta_{k+1} = gamma_dec * Delta_k
            # ==============================================================
            else:
                delta *= gamma_dec
                valid_log.append(dict(k=k, inner=inner, branch="reject,shrink",
                                      mdl_valid=mdl_valid, delta=delta, chi=chi_k))
                _out = (x_k.copy(), d_before, s_norm, eval_f(x_k), Aset, None,
                        mv, f"reject: c{i_viol} violated" if i_viol
                        else "reject: s=0")
                break

        # one trace line per outer iteration
        if verbose and _out is not None:
            _x, _d, _sn, _f, _A, _r, _mv, _br = _out
            if _grew:
                _br = f"{_br}   [A_k grew: {','.join(_grew)}]"
            row(k, crit, inner, _x, _d, _sn, chi_k, _f, _A, _r, _mv, _br)

        trace_active.append(list(A_k))
        k += 1

    # ======================================================================
    # WRAP-UP
    # ======================================================================
    if verbose: print("-" * len(hdr))
    print(f"\nFINAL x={np.array2string(x_k, precision=4, suppress_small=True)}  "
          f"f={f(x_k):.5f}   chi={chi_k:.3e}   "
          + "  ".join(f"c{i+1}={constraints[i](x_k):+.3f}" for i in range(p)))
    print(f"f-evals={evf[0]}  c-evals={evc[0]}  total={evf[0]+evc[0]}  "
          f"outer={k}  GROWTH events={growth}")

    # --- [ADDED A1] the four exits, only the first certifying --------------
    #   1  converged   chi_k <= chi_tol AND delta <= delta_term   <- certifies
    #   2  delta floor delta < delta_min                          <- does not
    #   3  budget      nf_max evaluations reached                 <- does not
    #   4  model       formquad returned no gradient              <- does not
    if not terminated:                       # fell out of the outer while
        term_reason = (f"budget: reached nf_max={nf_max} evaluations "
                       f"(NOT certified stationary)")
    print(f"\n[termination]  {term_reason}")

    # --- model-validity report (observation only; no decision used it) -----
    # Assumption assm:interpolation requires the models to be fully linear.  A
    # certificate issued while `valid` was False is nominal, not proven, since
    # the lem.chis error bound does not apply.
    n_steps = len(valid_log)
    n_invalid = sum(1 for r in valid_log if not r["mdl_valid"])
    print(f"[validity]  steps logged={n_steps}  invalid-model steps={n_invalid}"
          f"  ({100.0*n_invalid/max(n_steps,1):.0f}%)")
    if converged_valid is not None:          # set only on the CONVERGED exit
        verdict = "VALID (trustworthy)" if converged_valid else "INVALID (SUSPECT solve)"
        print(f"[validity]  model at convergence: {verdict}")
    else:
        print(f"[validity]  no stationarity certificate issued")

    return {"trace_pts": np.array(trace_pts), "trace_active": trace_active,
            "overshoots": overshoots, "X": np.array(pts), "x_final": x_k,
            "chi_final": chi_k, "valid_log": valid_log, "term_reason": term_reason,
            "converged_valid": converged_valid,
            "evf": evf[0], "evc": evc[0], "outer": k, "growth": growth}


if __name__ == "__main__":
    # Smoke test.  min (x0-5)^2 + (x1-1)^2  s.t.  x0^2 <= 4, -0.5 <= x1 <= 4.
    # The bound x0 <= 2 is active at the solution, so x* = (2, 1), f* = 9.
    np.set_printoptions(precision=4, suppress=True)
    run(f=lambda x: (x[0]-5)**2 + (x[1]-1)**2,
        constraints=[lambda x: x[0]**2 - 4.0,
                     lambda x: -x[1] - 0.5,
                     lambda x: x[1] - 4.0],
        x0=[0.2, -0.4], delta0=1.0, delta_max=1.0)
    