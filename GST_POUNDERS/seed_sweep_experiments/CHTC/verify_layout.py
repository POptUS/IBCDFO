#!/usr/bin/env python
"""Prove a collected pilot directory is a drop-in replacement for the local
``all_methods_comparison/`` folder that ``analyze_all_methods.ipynb`` reads.

This does NOT check "looks similar". It replays the exact glob patterns and
column accesses the notebook performs, and fails on any that come back empty.
The reference inventory below was read off a real local run.

    python verify_layout.py by_pilot/pilot_0350

Exit status 0 means: point RESULTS_DIR at this directory and every cell that
worked locally will find its inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

INF = "mean_gate_entanglement_infidelity_to_truth"

# --- the reference, taken from all_methods_comparison/seed_010001/ --------------
FPR_FILES = {
    "adaptive_shot_events.csv", "completed.json", "config.json",
    "final_gate_errors.csv", "final_shots_per_circuit.npy", "final_spam_errors.csv",
    "fpr_selection_history.csv", "iteration_accuracy.csv", "iteration_gate_errors.csv",
    "iteration_spam_errors.csv", "optimizer_progress.csv", "problem_metadata.json",
    "summary.csv", "summary.json", "x_best.npy", "x_evaluations.npy",
}
LM_FILES = {"lm_trajectory.csv", "summary.json"}
SUMMARY_COLS = ["seed", "method", INF, "accounted_revealed_shots",
                "max_shots_per_circuit", "flag"]
LABELS = {"fixed_FPR", "no_FPR", "adaptive_D", "LM"}

ok, bad = [], []


def check(cond, msg, detail=""):
    (ok if cond else bad).append(msg)
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}{('  -- ' + detail) if detail and not cond else ''}")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    R = Path(sys.argv[1])
    if not R.is_dir():
        sys.exit(f"not a directory: {R}")
    print(f"checking {R.resolve()}\n")

    # ---- 1. the summary the notebook loads first ------------------------------
    csv = R / "all_methods_summary.csv"
    check(csv.exists(), "all_methods_summary.csv exists")
    if csv.exists():
        s = pd.read_csv(csv)
        check(list(s.columns) == SUMMARY_COLS, "summary columns match local exactly",
              f"got {list(s.columns)}")
        for c in ("seed", "method", INF):
            check(c in s.columns, f"summary has the column the notebook reads: {c!r}")
        found = set(s["method"].unique()) if "method" in s else set()
        check(found <= LABELS and found, f"method labels are notebook labels: {sorted(found)}",
              f"unexpected {sorted(found - LABELS)}")
        check(not s[INF].isna().all() if INF in s else False,
              "summary infidelities are not all NaN")

    # ---- 2. the notebook's per-run globs --------------------------------------
    seeds = sorted(R.glob("seed_*"))
    check(bool(seeds), f"seed_* directories present ({len(seeds)})")

    for fname in ("iteration_accuracy.csv", "optimizer_progress.csv",
                  "adaptive_shot_events.csv", "fpr_selection_history.csv",
                  "summary.json"):
        hits = list(R.glob(f"seed_*/*/{fname}"))
        check(bool(hits), f'notebook glob "seed_*/*/{fname}"  -> {len(hits)} hits')

    lm = list(R.glob("seed_*/lm/lm_trajectory.csv"))
    check(bool(lm), f'notebook glob "seed_*/lm/lm_trajectory.csv" -> {len(lm)} hits')

    # ---- 3. per-method inventory vs the local reference -----------------------
    for sd in seeds:
        folders = {p.name for p in sd.iterdir() if p.is_dir() and not p.name.startswith(".")}
        check(folders >= {"fixed_fpr", "fixed_no_fpr", "adaptive_D", "lm"},
              f"{sd.name}: all four method folders", f"got {sorted(folders)}")
        for f in sorted(folders):
            have = {p.name for p in (sd / f).iterdir() if p.is_file()}
            want = LM_FILES if f == "lm" else FPR_FILES
            check(have >= want, f"{sd.name}/{f}: {len(want)} reference files present",
                  f"missing {sorted(want - have)}")

    # ---- 4. the columns the plotting cells need -------------------------------
    for fname, cols in (("iteration_accuracy.csv", ["iteration", INF]),
                        ("optimizer_progress.csv", ["nf", "ng"])):
        hits = list(R.glob(f"seed_*/*/{fname}"))
        if hits:
            c = set(pd.read_csv(hits[0], nrows=1).columns)
            for col in cols:
                check(col in c, f"{fname} has column {col!r} (needed by the plots)",
                      f"has {sorted(c)[:8]}...")

    print(f"\n{len(ok)} passed, {len(bad)} failed")
    if bad:
        print("\nFAILURES:")
        for b in bad:
            print("  -", b)
        return 1
    print(f"\nOK -- in analyze_all_methods.ipynb set:\n    RESULTS_DIR = Path(r\"{R.resolve()}\")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
