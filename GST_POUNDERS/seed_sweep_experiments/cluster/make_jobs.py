#!/usr/bin/env python3
"""Write joblist.txt -- one line per HTCondor job: "<seed>, <extra run_one.py args>".

Two modes.

FULL SWEEP (default): every arm, every seed, budget anchored by fixed_fpr inside the job.

    python make_jobs.py

ONE ARM AGAINST AN EXISTING RUN (--from-results): run only the arms you name, reusing the
budget anchor and the settings of a sweep you already have. Use this to add a variant without
paying for the arms that would not change:

    python make_jobs.py --from-results ../all_methods_comparison_1000 \
        --arms adaptive_D_nofpr --fracs 0.1,0.2,0.3,0.4,0.5,0.6

That reads each seed's fixed_fpr anchor and pins --budget to it, and copies nfmax /
allocate-every / objective / criterion off the existing arm so the new runs are directly
comparable. Each fraction gets its own output folder (adaptive_D_nofpr_f030, ...), so the
whole curve lands in one results tree and the notebooks plot each as its own series.

WHY THE ANCHOR MUST BE COPIED: the budget is fixed_fpr's accounted revealed-shot count and it
differs per seed (819,000 / 963,000 / 836,000 in the _1000 run). Re-deriving it would need
fixed_fpr to run again; pinning it costs nothing and keeps the comparison matched.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

# ----------------------------------------------------------------- full-sweep defaults
SEEDS = list(range(101, 121))

ARGS = [
    "--objective poisson_logl --allocate-every 5 --nfmax 150",
]

# Settings copied from the reference run in --from-results mode, and the run_one.py flag
# each maps to. Anything not listed is left at the config default.
_COPY = {
    "objective": "--objective",
    "nfmax": "--nfmax",
    "adaptive_allocate_every": "--allocate-every",
}


def from_results(ref_dir, arms, fracs, seeds=None, extra=""):
    ref = Path(ref_dir)
    seed_dirs = sorted(d for d in ref.glob("seed_*") if d.is_dir())
    if not seed_dirs:
        sys.exit(f"no seed_* directories under {ref}")
    lines, skipped = [], []
    settings_seen = set()
    for sd in seed_dirs:
        seed = int(sd.name.split("_")[1])
        if seeds and seed not in seeds:
            continue
        anchor_f = sd / "fixed_fpr" / "summary.json"
        if not anchor_f.exists():
            skipped.append((seed, "no fixed_fpr/summary.json -- cannot recover the budget"))
            continue
        budget = int(json.loads(anchor_f.read_text())["accounted_revealed_shots"])

        # copy settings off any completed arm in this seed, so the new runs match
        cfg = None
        for cand in sorted(sd.iterdir()):
            f = cand / "config.json"
            if f.exists():
                cfg = json.loads(f.read_text())
                break
        if cfg is None:
            skipped.append((seed, "no config.json in any arm"))
            continue
        copied = " ".join(f"{flag} {cfg[key]}" for key, flag in _COPY.items() if key in cfg)
        settings_seen.add(copied)

        for arm, frac in itertools.product(arms, fracs):
            suffix = f"_f{int(round(frac * 100)):03d}"
            # `extra` goes AFTER the copied settings: argparse keeps the last occurrence
            # of a flag, so --extra "--nfmax 500" overrides a copied --nfmax 300.
            args = (f"--arms {arm} --budget {budget} {copied} {extra} "
                    f"--nofpr-baseline-frac {frac} --label-suffix {suffix}").replace("  ", " ")
            assert "," not in args, f"no commas allowed: {args!r}"
            lines.append(f"{seed}, s{seed}_{arm}{suffix}, {args}")

    if len(settings_seen) > 1:
        print("WARNING: the reference seeds do not share one setting set:", flush=True)
        for s in sorted(settings_seen):
            print(f"    {s}")
        print("  The new runs will inherit each seed's own settings, which is right, but the")
        print("  reference arms are not comparable ACROSS seeds. Check before pooling.")
    elif settings_seen:
        print(f"copied settings: {next(iter(settings_seen))}")
    for seed, why in skipped:
        print(f"  skipped seed {seed}: {why}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="joblist.txt")
    ap.add_argument("--from-results", default=None,
                    help="an existing results tree to take budgets and settings from")
    ap.add_argument("--arms", default="adaptive_D_nofpr",
                    help="--from-results mode: comma list of arms to run")
    ap.add_argument("--fracs", default="0.1,0.2,0.3,0.4,0.5,0.6",
                    help="--from-results mode: baseline fractions to sweep")
    ap.add_argument("--seeds", default=None,
                    help="restrict to these seeds, e.g. 101,102,103")
    ap.add_argument("--extra", default="",
                    help="extra run_one.py flags, appended AFTER the copied settings so they "
                         "override them, e.g. --extra \"--nfmax 500\". Anything you override "
                         "here no longer matches the reference arms -- fine for a flag they "
                         "do not use (LM ignores nfmax), confounding for one they do.")
    a = ap.parse_args()

    seeds = {int(s) for s in a.seeds.split(",")} if a.seeds else None

    if a.from_results:
        lines = from_results(a.from_results,
                             [m.strip() for m in a.arms.split(",") if m.strip()],
                             [float(f) for f in a.fracs.split(",")],
                             seeds, a.extra.strip())
        if a.extra.strip():
            print(f"overrides applied after the copied settings: {a.extra.strip()}")
            print("  -> POUNDERS arms in the reference run do NOT share these. Compare the "
                  "new arms against lm/lm_mle/lm_fpr (which ignore nfmax), and treat "
                  "fixed_fpr / fixed_no_fpr / adaptive_D as not directly matched.")
    else:
        use = sorted(seeds) if seeds else SEEDS
        lines = []
        for i, (seed, args) in enumerate(itertools.product(use, ARGS)):
            assert "," not in args, f"no commas allowed in ARGS: {args!r}"
            # unique even when ARGS has several entries for the same seed
            tag = f"s{seed}" if len(ARGS) == 1 else f"s{seed}_c{i % len(ARGS)}"
            lines.append(f"{seed}, {tag}, {args}")

    if not lines:
        sys.exit("no jobs written -- nothing matched")
    # the tag names the returned tarball; a duplicate would silently lose a job's results
    tags = [l.split(",")[1].strip() for l in lines]
    assert len(set(tags)) == len(tags), "job tags must be unique"
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}: {len(lines)} jobs")
    for l in lines[:3]:
        print("   ", l)
    if len(lines) > 3:
        print(f"    ... and {len(lines) - 3} more")


if __name__ == "__main__":
    main()
