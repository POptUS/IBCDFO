#!/usr/bin/env python
"""Run one (pilot-shots, data-seed) cell of the sweep: every method, matched budget.

This mirrors the protocol in ``compare_all_methods.ipynb`` exactly:

  1. ``fixed_fpr`` runs first at ``--fixed-fpr-shots`` per circuit. Its accounted
     revealed-shot count is the *budget anchor* for everything else, unless
     ``--budget`` pins one explicitly.
  2. ``no_FPR``  = uniform over ALL circuits at ``budget // total_circuits``.
  3. ``adaptive_<crit>`` = adaptive FPR with ``adaptive_total_shot_budget = budget``.
  4. ``lm``     = pyGSTi StandardGST/LM on uniform shots at the same budget.

All methods for a seed MUST run in the same job, because the budget anchor is
produced by fixed_fpr. Splitting them across jobs would break the matching.

Writes ``<outdir>/seed_<seed>/<method>/`` plus a one-row-per-method ``result.csv``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import replace, asdict
from pathlib import Path

import numpy as np
import pandas as pd

INF = "mean_gate_entanglement_infidelity_to_truth"
SPAM = "mean_spam_vector_l2_error_to_truth"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", type=int, required=True,
                    help="adaptive_baseline_shots (the 'pilot shots' being swept)")
    ap.add_argument("--seed", type=int, required=True, help="data seed")
    ap.add_argument("--config", default="experiment_config.json")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--methods", default="fixed_fpr,fixed_no_fpr,adaptive_D,lm",
                    help="comma list; adaptive_<C> for criterion C")
    ap.add_argument("--fixed-fpr-shots", type=int, default=None,
                    help="override config.fixed_fpr_shots (the budget anchor)")
    ap.add_argument("--budget", type=int, default=None,
                    help="pin the total shot budget instead of taking fixed_fpr's "
                         "accounted cost. STRONGLY recommended for a pilot sweep, so "
                         "that pilot shots are the only variable across cells.")
    ap.add_argument("--nfmax", type=int, default=None, help="override config.nfmax")
    ap.add_argument("--spam-noise", type=float, default=None,
                    help="override config.spam_noise on the TRUTH model. NB 0.0 gives "
                         "ideal SPAM, which puts 16 outcomes at truth probability "
                         "exactly 0 or 1; those never produce an informative count at "
                         "any shot budget. Keep --variance-smoothing on if you use it.")
    ap.add_argument("--variance-smoothing", type=float, default=None,
                    help="add-s smoothing on the WLS variance. 0.0 = historical "
                         "behaviour (var can be 0 -> weight 1/variance_floor = 1e12); "
                         "0.5 = Jeffreys, bounded away from 0 so the floor never binds.")
    ap.add_argument("--lm-maxiter", type=int, default=800)
    ap.add_argument("--lm-modes", default="CPTPLND")
    return ap.parse_args()


def build_lm(cfg, seed, target_shots, lm_dir, GSTProblem, lm_modes, lm_maxiter):
    """pyGSTi LM on uniform shots at the matched budget (port of the notebook cell)."""
    import pygsti
    from pygsti.optimize import SimplerLMOptimizer

    prob = GSTProblem(cfg, seed)
    try:
        prob.base_model.sim = "map"
        prob.truth_model.sim = "map"
    except Exception:
        pass
    per = max(1, round(target_shots / len(prob.circuits)))
    shots = prob.normalize_shots(per)
    dataset = prob.simulate_dataset(shots, seed=9_000_000 + seed)
    data = pygsti.protocols.ProtocolData(prob.design, dataset)
    opt = SimplerLMOptimizer(maxiter=lm_maxiter, maxfev=lm_maxiter, tol=1e-6,
                             init_munu="auto", oob_action="reject")
    proto = pygsti.protocols.StandardGST(modes=lm_modes, target_model=prob.target_model,
                                         optimizer=opt, verbosity=0)
    res = proto.run(data)
    keys = list(res.estimates.keys())
    est = res.estimates[lm_modes if lm_modes in res.estimates else keys[0]]
    fit = est.models["final iteration estimate"]
    summ, _, _ = prob.aligned_error_metrics(fit, prob.truth_model, "truth")

    mls = list(getattr(cfg, "max_lengths", []))
    stage_keys = sorted([k for k in est.models if re.fullmatch(r"iteration \d+ estimate", k)],
                        key=lambda k: int(k.split()[1]))
    traj = []
    for si, k in enumerate(stage_keys):
        try:
            st, _, _ = prob.aligned_error_metrics(est.models[k], prob.truth_model, "truth")
            traj.append({"stage": si, "max_length": (mls[si] if si < len(mls) else si),
                         INF: float(st[INF]), SPAM: float(st.get(SPAM, float("nan")))})
        except Exception:
            pass

    out = {"data_seed": seed, "method": "lm", INF: float(summ[INF]),
           SPAM: float(summ.get(SPAM, float("nan"))),
           "accounted_revealed_shots": int(shots.sum()),
           "physical_precomputed_shots": int(shots.sum()),
           "max_shots_per_circuit": int(shots.max()),
           "min_shots_per_circuit": int(shots.min()),
           "mean_shots_per_circuit": float(shots.mean()),
           "total_circuits": int(len(prob.circuits)),
           "revealed_circuits": int(len(prob.circuits)), "flag": "LM"}
    lm_dir.mkdir(parents=True, exist_ok=True)
    (lm_dir / "summary.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(traj).to_csv(lm_dir / "lm_trajectory.csv", index=False)
    return out


def main():
    args = parse_args()

    # single-threaded BLAS: heterogeneous execute nodes otherwise make runs
    # irreproducible, and a single marginal accept/reject decides the outcome here.
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")

    from gst_seed_experiment import ExperimentConfig, GSTProblem, run_one_experiment

    base = ExperimentConfig.from_json(args.config)
    over = {"adaptive_baseline_shots": int(args.pilot)}
    if args.fixed_fpr_shots is not None:
        over["fixed_fpr_shots"] = int(args.fixed_fpr_shots)
    if args.nfmax is not None:
        over["nfmax"] = int(args.nfmax)
    if args.variance_smoothing is not None:
        over["variance_smoothing"] = float(args.variance_smoothing)
    if args.spam_noise is not None:
        over["spam_noise"] = float(args.spam_noise)
    base = replace(base, **over)

    outdir = Path(args.outdir)
    seed_dir = outdir / f"seed_{args.seed:06d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    rows, t0 = [], time.time()
    print(f"[run_one] pilot={args.pilot} seed={args.seed} methods={methods}", flush=True)

    # ---- 1. the budget anchor -------------------------------------------------
    budget = args.budget
    total_circuits = None
    if "fixed_fpr" in methods or budget is None:
        t = time.time()
        fixed = run_one_experiment(config=base, data_seed=args.seed, method="fixed_fpr",
                                   output_dir=seed_dir / "fixed_fpr")
        total_circuits = int(fixed["total_circuits"])
        if budget is None:
            budget = int(fixed["accounted_revealed_shots"])
        rows.append(dict(pilot=args.pilot, seed=args.seed, method="fixed_FPR",
                         secs=round(time.time() - t, 1), **_pick(fixed)))
        print(f"[run_one] fixed_fpr done, budget anchor = {budget:,}", flush=True)
    if total_circuits is None:                      # --budget given and fixed_fpr skipped
        total_circuits = len(GSTProblem(base, args.seed).circuits)

    # ---- 2. no_FPR: uniform over every circuit at the same budget -------------
    if "fixed_no_fpr" in methods:
        t = time.time()
        cfg = replace(base, fixed_no_fpr_shots=max(1, budget // total_circuits))
        nf = run_one_experiment(config=cfg, data_seed=args.seed, method="fixed_no_fpr",
                                output_dir=seed_dir / "fixed_no_fpr")
        rows.append(dict(pilot=args.pilot, seed=args.seed, method="no_FPR",
                         secs=round(time.time() - t, 1), **_pick(nf)))
        print("[run_one] fixed_no_fpr done", flush=True)

    # ---- 3. adaptive criteria -------------------------------------------------
    for m in methods:
        if not m.startswith("adaptive_"):
            continue
        crit = m.split("_", 1)[1]
        t = time.time()
        cfg = replace(base, adaptive_criterion=crit, adaptive_total_shot_budget=int(budget))
        try:
            res = run_one_experiment(config=cfg, data_seed=args.seed, method="adaptive_fpr",
                                     output_dir=seed_dir / f"adaptive_{crit}")
            rows.append(dict(pilot=args.pilot, seed=args.seed, method=f"adaptive_{crit}",
                             secs=round(time.time() - t, 1), **_pick(res)))
        except Exception:
            traceback.print_exc()
            rows.append(dict(pilot=args.pilot, seed=args.seed, method=f"adaptive_{crit}",
                             secs=round(time.time() - t, 1), **_pick({}), flag="FAILED"))
        print(f"[run_one] adaptive_{crit} done", flush=True)

    # ---- 4. LM ----------------------------------------------------------------
    if "lm" in methods:
        t = time.time()
        try:
            lm = build_lm(base, args.seed, budget, seed_dir / "lm", GSTProblem,
                          args.lm_modes, args.lm_maxiter)
            rows.append(dict(pilot=args.pilot, seed=args.seed, method="LM",
                             secs=round(time.time() - t, 1), **_pick(lm)))
        except Exception:
            traceback.print_exc()
            rows.append(dict(pilot=args.pilot, seed=args.seed, method="LM",
                             secs=round(time.time() - t, 1), **_pick({}), flag="FAILED"))
        print("[run_one] lm done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(seed_dir / "result.csv", index=False)
    (seed_dir / "cell_meta.json").write_text(json.dumps(
        {"pilot": args.pilot, "seed": args.seed, "budget": int(budget),
         "total_circuits": int(total_circuits), "methods": methods,
         "elapsed_secs": round(time.time() - t0, 1),
         "config": json.loads(json.dumps(asdict(base), default=str))}, indent=2))
    print(df.to_string(index=False), flush=True)
    print(f"[run_one] TOTAL {time.time() - t0:.0f}s", flush=True)


def _pick(s):
    return {INF: s.get(INF, float("nan")),
            SPAM: s.get(SPAM, float("nan")),
            "accounted_revealed_shots": s.get("accounted_revealed_shots"),
            "physical_precomputed_shots": s.get("physical_precomputed_shots"),
            "max_shots_per_circuit": s.get("max_shots_per_circuit"),
            "flag": s.get("flag")}


if __name__ == "__main__":
    sys.exit(main())
