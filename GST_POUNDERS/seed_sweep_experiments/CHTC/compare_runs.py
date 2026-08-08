#!/usr/bin/env python
"""Compare two sweep_summary.csv files, and rank the methods within the new one.

    ./rolenv/bin/python compare_runs.py ~/run1_baseline/sweep_summary.csv sweep_summary.csv

Everything is paired by seed: the same data seed, the same shot budget, the two
runs differing only in the setting under test. Unpaired medians are not a fair
comparison when one method is bimodal, which is the trap that made adaptive_D
look like it beat LM locally.

A method whose values are IDENTICAL across the two runs is reported as such
rather than crashing -- for a change that only touches the WLS objective, LM
coming back bit-identical is the control working, not an error.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

INF = "mean_gate_entanglement_infidelity_to_truth"
ORDER = ["LM", "fixed_FPR", "no_FPR", "adaptive_D"]


def paired(x: pd.Series, y: pd.Series):
    """Paired stats for y vs x. Returns (n, ratio_of_medians, wins, p, note)."""
    i = x.index.intersection(y.index)
    x, y = x.loc[i].astype(float), y.loc[i].astype(float)
    d = (y - x).to_numpy()
    if len(i) == 0:
        return 0, np.nan, 0, np.nan, "no common seeds"
    if np.all(d == 0):
        return len(i), 1.0, 0, np.nan, "IDENTICAL on every seed"
    r = (y / x).to_numpy()
    p = stats.wilcoxon(x, y).pvalue if len(i) > 5 else np.nan
    return len(i), float(np.median(r)), int((r < 1).sum()), p, ""


def load(path, pilot=None):
    d = pd.read_csv(path)
    if pilot is not None and "pilot" in d.columns:
        d = d[d["pilot"] == pilot]
    return {m: g.set_index("seed")[INF] for m, g in d.groupby("method")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new")
    ap.add_argument("--pilot", type=int, default=500,
                    help="restrict the OLD file to this pilot (default 500)")
    a = ap.parse_args()

    old, new = load(a.old, a.pilot), load(a.new)

    print(f"=== effect of the change, paired by seed  ({a.old} -> {a.new}) ===")
    print(f"{'method':<12}{'n':>4}{'old':>12}{'new':>12}{'ratio':>8}{'better':>8}{'p':>9}   note")
    for m in ORDER:
        if m not in old or m not in new:
            continue
        n, r, wins, p, note = paired(old[m], new[m])
        i = old[m].index.intersection(new[m].index)
        ps = "  ident." if np.isnan(p) and note else f"{p:9.4f}"
        print(f"{m:<12}{n:>4}{old[m].loc[i].median():12.3e}{new[m].loc[i].median():12.3e}"
              f"{r:8.3f}{wins:>8}{ps}   {note}")
    print("\n  ratio < 1 means the change IMPROVED that method; 'better' counts seeds improved.")

    print(f"\n=== method ranking in the NEW run, paired head-to-head ===")
    med = {m: v.median() for m, v in new.items() if m in ORDER}
    for m, v in sorted(med.items(), key=lambda kv: kv[1]):
        print(f"    {m:<12}{v:.3e}")
    print(f"\n{'pair':<26}{'n':>4}{'ratio':>8}{'wins':>7}{'p':>9}")
    for x, y in itertools.combinations([m for m in ORDER if m in new], 2):
        n, r, wins, p, note = paired(new[x], new[y])
        ps = "   ident." if np.isnan(p) else f"{p:9.4f}"
        print(f"{y+' / '+x:<26}{n:>4}{r:8.3f}{wins:>7}{ps}   {note}")
    print(f"\n  ratio < 1 means the FIRST-named method is better; p < 0.05 means it is not chance.")


if __name__ == "__main__":
    main()
