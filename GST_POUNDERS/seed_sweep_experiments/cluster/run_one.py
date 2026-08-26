#!/usr/bin/env python3
"""Run ONE data seed: every arm, matched budget. This is the unit of work on the cluster.

Mirrors compare_all_methods.ipynb exactly:

  1. fixed_fpr    POUNDERS + FPR, uniform shots. Its accounted revealed-shot count is the
                  BUDGET ANCHOR for every other arm, unless --budget pins one.
  2. fixed_no_fpr POUNDERS, uniform over ALL circuits at budget // total_circuits.
  3. adaptive_<C>        POUNDERS + FPR + adaptive allocation, criterion C.
     adaptive_<C>_nofpr  the same allocation with FPR OFF, so the allocation gain is not
                         bundled with the circuit-reduction loss. Its uniform baseline is
                         derived from the budget (--nofpr-baseline-frac), because with no
                         mask every circuit is billed and the configured per-circuit
                         adaptive_baseline_shots would blow the budget at iteration 0.
  4. lm, lm_mle, lm_fpr  the pyGSTi Levenberg-Marquardt baselines (see lm_arms.py).

Every arm for a seed MUST run in the same job: the budget anchor is shared, and all arms
draw from the same data seed so they are one realisation of one experiment.

Writes <outdir>/seed_<seed>/<arm>/ plus a one-row-per-arm result.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

# Pin BLAS to one thread BEFORE numpy is imported. Cluster nodes are heterogeneous, and in
# this problem a single marginal accept/reject decides which basin a seed lands in --
# multithreaded BLAS would make runs irreproducible across machines and you could not tell a
# real effect from numerical jitter. Also, HTCondor gives the job one core.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True, help="data seed")
    ap.add_argument("--config", default="experiment_config.json")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--arms",
                    default="fixed_fpr,fixed_no_fpr,adaptive_D,adaptive_D_nofpr,"
                            "lm,lm_mle,lm_fpr",
                    help="comma list. adaptive_<C> and adaptive_<C>_nofpr for criterion C.")
    ap.add_argument("--budget", type=int, default=None,
                    help="pin the total shot budget instead of taking fixed_fpr's anchor. "
                         "Required if fixed_fpr is not in --arms.")
    ap.add_argument("--fixed-fpr-shots", type=int, default=None,
                    help="override config.fixed_fpr_shots (the anchor's shots/circuit)")
    ap.add_argument("--nofpr-baseline-frac", type=float, default=0.5,
                    help="fraction of the budget the *_nofpr arms spend on their uniform "
                         "baseline; the rest is adaptively allocated. None-> use the config "
                         "value verbatim (only sensible if you have lowered it).")
    # plain config overrides, so a sweep axis needs no new code
    ap.add_argument("--pilot", type=int, default=None,
                    help="override config.adaptive_baseline_shots (the FPR arms' pilot)")
    ap.add_argument("--nfmax", type=int, default=None)
    ap.add_argument("--schedule-n-max", type=int, default=None,
                    help="override config.adaptive_schedule_n_max, the per-round cap on "
                         "allocated shots. The schedule is budget-BLIND -- n0, base, constant "
                         "and this cap are absolute -- so its total request is fixed at about "
                         "2.2M shots regardless of the budget. The hook clips a round to the "
                         "remaining budget but never pads one, so below that total the budget "
                         "is spent exactly and above it the remainder is silently left "
                         "unspent (measured: 79% of a 4000 budget, 54% of a 6000 budget). "
                         "Raise this for budgets whose adaptive remainder exceeds ~2.2M.")
    ap.add_argument("--allocate-every", type=int, default=None)
    ap.add_argument("--objective", default=None,
                    help="weighted_least_squares | least_squares | poisson_logl")
    ap.add_argument("--lm-maxiter", type=int, default=800)
    ap.add_argument("--lm-modes", default="CPTPLND")
    ap.add_argument("--lm-data-seed-offset", type=int, default=0)
    ap.add_argument("--fpr-cache", default="fpr_builtin",
                    help="where the pyGSTi built-in FPR pair set is cached")
    ap.add_argument("--label-suffix", default="",
                    help="appended to each arm's OUTPUT FOLDER name (not the method). Lets "
                         "several settings of the same arm live in one results tree, e.g. "
                         "--nofpr-baseline-frac 0.3 --label-suffix _f030 writes "
                         "adaptive_D_nofpr_f030/. The analysis notebooks discover arms by "
                         "folder name, so each variant plots as its own series.")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    t_start = time.time()

    from gst_seed_experiment import ExperimentConfig, GSTProblem, run_one_experiment
    import lm_arms

    base = ExperimentConfig.from_json(a.config)
    over = {}
    if a.pilot is not None:
        over["adaptive_baseline_shots"] = int(a.pilot)
    if a.nfmax is not None:
        over["nfmax"] = int(a.nfmax)
    if a.allocate_every is not None:
        over["adaptive_allocate_every"] = int(a.allocate_every)
    if a.schedule_n_max is not None:
        over["adaptive_schedule_n_max"] = int(a.schedule_n_max)
    if a.objective is not None:
        over["objective"] = a.objective
    if over:
        base = replace(base, **over)
        print(f"[run_one] config overrides: {over}", flush=True)

    arms = [m.strip() for m in a.arms.split(",") if m.strip()]
    seed_dir = Path(a.outdir) / f"seed_{a.seed:06d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(name, summary, seconds, error=None):
        rows.append(dict(seed=a.seed, arm=name,
                         infidelity=summary.get(lm_arms.INF) if summary else None,
                         diamond=summary.get(lm_arms.DD) if summary else None,
                         spam=summary.get(lm_arms.SPAM) if summary else None,
                         accounted_revealed_shots=(summary or {}).get(
                             "accounted_revealed_shots"),
                         revealed_circuits=(summary or {}).get("revealed_circuits"),
                         total_circuits=(summary or {}).get("total_circuits"),
                         objective=(summary or {}).get("objective_mode")
                                   or (summary or {}).get("objective"),
                         flag=(summary or {}).get("flag"),
                         seconds=round(seconds, 1), error=error))
        pd.DataFrame(rows).to_csv(seed_dir / "result.csv", index=False)

    # ---------------------------------------------------------------- 1. the budget anchor
    budget = a.budget
    total_circuits = None
    if "fixed_fpr" in arms:
        cfg = base if a.fixed_fpr_shots is None else replace(
            base, fixed_fpr_shots=int(a.fixed_fpr_shots))
        t0 = time.time()
        print(f"[run_one] seed={a.seed} fixed_fpr (anchor)", flush=True)
        try:
            s = run_one_experiment(config=cfg, data_seed=a.seed, method="fixed_fpr",
                                   output_dir=seed_dir / "fixed_fpr")
        except Exception as exc:
            # The anchor failing is fatal only when nothing pinned the budget: every other
            # arm is sized from it. With --budget given, carry on and report the failure.
            traceback.print_exc()
            record("fixed_fpr", None, time.time() - t0, error=repr(exc))
            if budget is None:
                sys.exit(f"[run_one] fixed_fpr is the budget anchor and it failed "
                         f"({exc!r}); pass --budget to run the other arms anyway")
            s = None
        if s is not None:
            record("fixed_fpr", s, time.time() - t0)
            anchor = int(s["accounted_revealed_shots"])
            total_circuits = int(s["total_circuits"])
            if budget is None:
                budget = anchor
            print(f"[run_one] anchor = {anchor:,} accounted shots "
                  f"(budget in use: {budget:,})", flush=True)
    if budget is None:
        sys.exit("--budget is required when fixed_fpr is not in --arms")

    if total_circuits is None:                      # cheap probe, only if we skipped the anchor
        total_circuits = len(GSTProblem(base, a.seed).circuits)

    # ------------------------------------------------------------------ 2. uniform, no FPR
    if "fixed_no_fpr" in arms:
        t0 = time.time()
        per = max(1, budget // total_circuits)
        print(f"[run_one] seed={a.seed} fixed_no_fpr ({per} sh/circ)", flush=True)
        try:
            s = run_one_experiment(config=replace(base, fixed_no_fpr_shots=per),
                                   data_seed=a.seed, method="fixed_no_fpr",
                                   output_dir=seed_dir / "fixed_no_fpr")
            record("fixed_no_fpr", s, time.time() - t0)
        except Exception as exc:
            traceback.print_exc()
            record("fixed_no_fpr", None, time.time() - t0, error=repr(exc))

    # -------------------------------------------------------------------- 3. adaptive arms
    for arm in [m for m in arms if m.startswith("adaptive_")]:
        nofpr = arm.endswith("_nofpr")
        crit = arm[len("adaptive_"):-len("_nofpr")] if nofpr else arm[len("adaptive_"):]
        method = "adaptive_no_fpr" if nofpr else "adaptive_fpr"
        cfg = replace(base, adaptive_criterion=crit,
                      adaptive_total_shot_budget=int(budget))
        if nofpr and a.nofpr_baseline_frac is not None:
            # every circuit is billed here, so scale the baseline to the budget
            b = max(1, int(round(a.nofpr_baseline_frac * budget / max(total_circuits, 1))))
            cfg = replace(cfg, adaptive_baseline_shots=b)
            print(f"[run_one] {arm}: baseline {b} sh/circ x {total_circuits} = "
                  f"{b * total_circuits:,} ({b * total_circuits / budget:.0%} of budget)",
                  flush=True)
        t0 = time.time()
        label = arm + a.label_suffix
        print(f"[run_one] seed={a.seed} {label} (method={method})", flush=True)
        try:
            s = run_one_experiment(config=cfg, data_seed=a.seed, method=method,
                                   output_dir=seed_dir / label)
            record(label, s, time.time() - t0)
        except Exception as exc:
            traceback.print_exc()
            record(label, None, time.time() - t0, error=repr(exc))

    # -------------------------------------------------------------------------- 4. LM arms
    lm_arms_wanted = [m for m in arms if m.startswith("lm")]
    if lm_arms_wanted:
        cfg = replace(base, report_diamond_distance=base.report_diamond_distance)
        prob = GSTProblem(cfg, a.seed)
        fpr_design = None
        for arm in lm_arms_wanted:
            t0 = time.time()
            print(f"[run_one] seed={a.seed} {arm}", flush=True)
            try:
                if arm == "lm":
                    s = lm_arms.fit_lm(prob, cfg, a.seed, budget, seed_dir / "lm",
                                       maxiter=a.lm_maxiter, modes=a.lm_modes,
                                       data_seed_offset=a.lm_data_seed_offset)
                elif arm == "lm_mle":
                    s = lm_arms.fit_lm_mle(prob, cfg, a.seed, budget, seed_dir / "lm_mle",
                                           maxiter=a.lm_maxiter,
                                           data_seed_offset=a.lm_data_seed_offset)
                elif arm == "lm_fpr":
                    if fpr_design is None:
                        fpr_design = lm_arms.builtin_fpr_design(prob, cfg, a.fpr_cache)
                    s = lm_arms.fit_lm_fpr(prob, cfg, a.seed, budget, seed_dir / "lm_fpr",
                                           fpr_design, maxiter=a.lm_maxiter,
                                           data_seed_offset=a.lm_data_seed_offset)
                else:
                    print(f"[run_one] unknown LM arm {arm!r}, skipped", flush=True)
                    continue
                record(arm, s, time.time() - t0)
            except Exception as exc:
                traceback.print_exc()
                record(arm, None, time.time() - t0, error=repr(exc))

    df = pd.DataFrame(rows)
    df.to_csv(seed_dir / "result.csv", index=False)
    (seed_dir / "run_metadata.json").write_text(json.dumps(
        {"seed": a.seed, "budget": int(budget), "total_circuits": int(total_circuits),
         "arms": arms, "args": vars(a), "seconds": round(time.time() - t_start, 1)},
        indent=2, default=str))
    print(f"\n[run_one] seed {a.seed} done in {time.time() - t_start:.0f}s")
    print(df.to_string(index=False))
    return 0 if df["error"].isna().all() else 1


if __name__ == "__main__":
    sys.exit(main())
