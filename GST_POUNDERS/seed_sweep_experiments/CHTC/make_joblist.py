#!/usr/bin/env python
"""Write joblist.txt -- one line per (pilot, seed) cell.

TEST 1: vary the pilot shots per circuit from 200 to 700 and see which wins.

Each job runs EVERY method for its seed, so the matched-budget protocol stays
intact. The total budget is pinned in sweep.sub (SWEEP_ARGS=--budget ...), so the
pilot is the only variable across cells.

Why the budget must be pinned, and why 550,000:
  budget = fixed_fpr_shots x (circuits FPR reveals), and the revealed count varies
  533-755 across seeds -- a 42% swing. Left unpinned, changing the pilot changes
  the trajectory, which changes what FPR reveals, which changes the budget, and
  the result is unattributable.

  550,000 is inside the observed anchor range (426k-604k) and leaves every seed
  some steerable budget even at pilot 700. At 465,600 (the median anchor), pilot
  700 would exhaust the whole budget on baseline alone for 3 of 13 seeds, making
  those cells degenerate -- adaptive would be exactly uniform.

Steerable fraction at budget 550,000, over the observed reveal range:
     pilot 200 ->  73-81%      pilot 500 ->  31-52%
     pilot 300 ->  59-71%      pilot 600 ->  18-42%
     pilot 400 ->  45-61%      pilot 700 ->   4-32%
"""
import itertools
import pathlib

PILOTS = [200, 300, 400, 500, 600, 700]
SEEDS = list(range(20001, 20021))          # 20 fresh seeds

lines = [f"{p} {s}" for p, s in itertools.product(PILOTS, SEEDS)]
pathlib.Path("joblist.txt").write_text("\n".join(lines) + "\n")

print(f"{len(lines)} jobs  ({len(PILOTS)} pilots x {len(SEEDS)} seeds)")
print(f"   pilots: {PILOTS}")
print(f"   seeds : {SEEDS[0]}..{SEEDS[-1]}")
print()
print("each job runs all methods for its seed (fixed_fpr, no_FPR, adaptive_D, LM)")
print("check sweep.sub has:  environment = \"SWEEP_ARGS=--budget 550000\"")
