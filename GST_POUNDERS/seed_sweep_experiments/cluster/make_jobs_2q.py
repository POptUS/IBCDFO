#!/usr/bin/env python3
"""Write the two-qubit joblists: ONE ARM PER JOB.

    python make_jobs_2q.py                     # defaults below
    python make_jobs_2q.py --shares 0.05 0.5   # two adaptive shares instead of three
    python make_jobs_2q.py --with-uniform      # add the uniform-POUNDERS baseline arm

Writes joblist_2q_pounders.txt and joblist_2q_lm.txt in this directory, and prints the job
count and a core-hour estimate.  Submit them separately so the fast LM arms are not stuck
behind the slow POUNDERS ones:

    condor_submit sweep_2q.sub -a "JOBLIST=joblist_2q_lm.txt"
    condor_submit sweep_2q.sub -a "JOBLIST=joblist_2q_pounders.txt"

Why one arm per job (this is the fix for the 72-hour hold, job 6151683): the old
joblist_2q.txt put `--arms adaptive_D_nofpr,lm,lm_mle` on one line, so all three ran in
sequence on one core.  POUNDERS alone used the whole 72 h and the LM arms never started.
Arms are independent as long as --budget is passed explicitly, which every line below does;
see the note at the top of run_one.py.

The --schedule-n-min value is the point of this script.  The shot schedule is budget-blind:
it asks for the same batch at iteration k whether the budget is 1.7M or 13.4M, and the hook
clips an over-request but never pads an under-request.  Left alone, a large budget is simply
never spent (at 2Q budget 250 the spend only completed at iteration 235 of 600).  A per-round
floor of (budget - baseline)/SPEND_BY spends it by iteration SPEND_BY instead, which is what
makes the arms budget-matched and lets nfmax come down to 300.
"""
import argparse
import math

# label -> total accounted shot budget.  Same five budgets as joblist_2q.txt.
BUDGETS = {250: 1_720_000, 500: 3_250_000, 1000: 6_860_000, 1500: 10_020_000, 2000: 13_350_000}
N_CIRCUITS_2Q = 13958          # StandardGSTDesign(smq2Q_XYICNOT, max_lengths 1..64)

# Rough per-arm wall clock on a CHTC node, for the estimate printed at the end.  POUNDERS:
# ~300 iterations x (Jacobian 103 s + trust-region solve + combine) plus ~60 allocator events
# with the factored-score allocator.  LM: measured 9 h (lm_mle) / 16 h (lm) at 2Q.
HOURS = {"pounders": 30.0, "lm": 16.0, "lm_mle": 9.0, "uniform": 20.0}


def lines_for(seed, label, budget, args, tag_suffix):
    return f"{seed}, s{seed}_b{label}_{tag_suffix}, {args}\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(101, 111)))
    ap.add_argument("--budgets", type=int, nargs="+", default=sorted(BUDGETS),
                    help="budget LABELS (see BUDGETS in this file)")
    ap.add_argument("--shares", type=float, nargs="+", default=[0.05, 0.30, 0.50],
                    help="uniform baseline FRACTIONS for the adaptive arms. 0.05 means 5%% of "
                         "the budget is spent uniformly and 95%% is allocated adaptively, "
                         "which the 1-qubit paper calls the 95%% adaptive share.")
    ap.add_argument("--nfmax", type=int, default=300)
    ap.add_argument("--allocate-every", type=int, default=5)
    ap.add_argument("--spend-by", type=int, default=150,
                    help="iteration by which the whole budget should be spent; sets the "
                         "per-round floor --schedule-n-min")
    ap.add_argument("--config", default="experiment_config_2q.json")
    ap.add_argument("--with-uniform", action="store_true",
                    help="also emit the uniform-POUNDERS arm (fixed_no_fpr). Needed only if "
                         "uniform POUNDERS becomes the Figure 5 baseline instead of LM.")
    ap.add_argument("--lm-arms", nargs="+", default=["lm", "lm_mle"])
    ap.add_argument("--pounders-out", default="joblist_2q_pounders.txt")
    ap.add_argument("--lm-out", default="joblist_2q_lm.txt")
    a = ap.parse_args()

    pounders, lm = [], []
    for label in a.budgets:
        budget = BUDGETS[label]
        for seed in a.seeds:
            for frac in a.shares:
                # run_one.py derives the same per-circuit baseline from --nofpr-baseline-frac;
                # recomputing it here is what makes the n_min arithmetic exact.
                per = max(1, round(frac * budget / N_CIRCUITS_2Q))
                baseline_total = per * N_CIRCUITS_2Q
                n_min = max(1, math.ceil((budget - baseline_total) / a.spend_by))
                suffix = f"_f{int(round(frac * 100)):03d}"
                args = (f"--config {a.config} --arms adaptive_D_nofpr --budget {budget} "
                        f"--objective poisson_logl --nfmax {a.nfmax} "
                        f"--allocate-every {a.allocate_every} --schedule-n-min {n_min} "
                        f"--nofpr-baseline-frac {frac} --label-suffix {suffix}")
                pounders.append(lines_for(seed, label, budget, args,
                                          f"pD{int(round(frac * 100)):03d}"))
            if a.with_uniform:
                args = (f"--config {a.config} --arms fixed_no_fpr --budget {budget} "
                        f"--objective poisson_logl --nfmax {a.nfmax}")
                pounders.append(lines_for(seed, label, budget, args, "unif"))
            for arm in a.lm_arms:
                args = f"--config {a.config} --arms {arm} --budget {budget}"
                lm.append(lines_for(seed, label, budget, args, arm.replace("_", "")))

    with open(a.pounders_out, "w", newline="\n") as fh:
        fh.writelines(pounders)
    with open(a.lm_out, "w", newline="\n") as fh:
        fh.writelines(lm)

    n_ad = len(a.seeds) * len(a.budgets) * len(a.shares)
    n_un = len(a.seeds) * len(a.budgets) if a.with_uniform else 0
    core_h = (n_ad * HOURS["pounders"] + n_un * HOURS["uniform"]
              + sum(HOURS.get(arm, 12.0) for arm in a.lm_arms) * len(a.seeds) * len(a.budgets))
    print(f"{a.pounders_out}: {len(pounders)} jobs "
          f"({n_ad} adaptive{f' + {n_un} uniform' if n_un else ''})")
    print(f"{a.lm_out}      : {len(lm)} jobs ({', '.join(a.lm_arms)})")
    print(f"seeds {a.seeds[0]}-{a.seeds[-1]}, budget labels {a.budgets}, shares {a.shares}")
    print(f"nfmax {a.nfmax}, allocate-every {a.allocate_every}, budget spent by iteration "
          f"{a.spend_by}")
    print(f"rough total ~{core_h:,.0f} core-hours "
          f"({HOURS['pounders']:.0f} h per POUNDERS arm, {HOURS['lm']:.0f}/{HOURS['lm_mle']:.0f} h per LM arm)")
    print("\nper-budget schedule floors (--schedule-n-min):")
    for label in a.budgets:
        budget = BUDGETS[label]
        row = []
        for frac in a.shares:
            per = max(1, round(frac * budget / N_CIRCUITS_2Q))
            row.append(f"f{int(round(frac * 100)):03d}={math.ceil((budget - per * N_CIRCUITS_2Q) / a.spend_by):,}")
        print(f"  b{label:<5d} budget {budget:>10,}   " + "   ".join(row))


if __name__ == "__main__":
    main()
