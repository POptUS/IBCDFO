#!/usr/bin/env python
"""Write joblist.txt -- one line per (pilot, seed) cell.

Each job runs EVERY method for its seed, because the matched-budget protocol
requires it: fixed_FPR runs first and its accounted cost is the budget the other
three are given. Splitting methods across jobs would break the matching.

WHY THE BUDGET IS NO LONGER PINNED
  sweep.sub used to pass --budget, which constrained no_FPR, adaptive_D and LM but
  NOT fixed_FPR -- fixed_FPR simply spent revealed_circuits x fixed_fpr_shots.
  Measured on run 3 that left it 13% over budget on 19 of 20 seeds (worst +66%),
  i.e. every comparison involving fixed_FPR was handing it extra shots.

  With --budget dropped, run_one.py falls back to its intended behaviour:

      budget = fixed_FPR's accounted_revealed_shots

  so all four methods match exactly within a seed. The budget then varies ACROSS
  seeds (541k-915k observed). Harmless here because every test is paired by seed,
  but do not compare raw medians between runs without pairing.

WHY ONE PILOT VALUE
  The pilot sweep was null on every test -- Kruskal p=0.92 across 200-700, chi2
  p=0.56 on the failure rate, best-vs-worst paired p=0.93. Re-sweeping it would
  cost 6x the jobs for nothing.

SWEEPING variance_smoothing
  Run it as two SEQUENTIAL submissions, changing only SWEEP_ARGS between them, and
  archive the first before starting the second:

      SWEEP_ARGS='--spam-noise 0.001 --variance-smoothing 0.01'
      SWEEP_ARGS='--spam-noise 0.001 --variance-smoothing 0.5'

  Then:  compare_runs.py <first>/sweep_summary.csv sweep_summary.csv --pilot 350

  Do NOT try to run both values in one submission: the returned tarball is named
  out_p<pilot>_s<seed>.tar.gz and cell_meta.json carries only pilot and seed, so the
  two cells would collide in by_pilot/ and silently overwrite each other.
"""
import itertools
import pathlib

PILOTS = [350]
SEEDS = list(range(20001, 20021))          # 20 seeds

lines = [f"{p} {s}" for p, s in itertools.product(PILOTS, SEEDS)]
pathlib.Path("joblist.txt").write_text("\n".join(lines) + "\n")

print(f"{len(lines)} jobs  ({len(PILOTS)} pilot x {len(SEEDS)} seeds)")
print(f"   pilots: {PILOTS}")
print(f"   seeds : {SEEDS[0]}..{SEEDS[-1]}")
print()
print("each job runs all methods for its seed (fixed_fpr, no_FPR, adaptive_D, LM)")
print("check sweep.sub has NO --budget, e.g.:")
print("   environment = \"SWEEP_ARGS='--spam-noise 0.001 --variance-smoothing 0.01'\"")
