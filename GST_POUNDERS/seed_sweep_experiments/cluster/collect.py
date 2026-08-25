#!/usr/bin/env python3
"""Unpack the tarballs HTCondor returned, build a summary table, and stage results for git.

    python collect.py                       # unpack ./out_*.tar.gz -> ./collected/
    python collect.py --stage ../results_chtc_run1

--stage writes a clean, git-friendly tree: one directory per seed, only the files the
analysis notebooks read. The excluded diagnostics were already dropped by job.sh, so this is
mostly a rename plus a size report -- it exists so you can eyeball the total before committing.
"""
import argparse
import glob
import json
import pathlib
import shutil
import sys
import tarfile

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=".", help="where out_*.tar.gz landed")
    ap.add_argument("--pattern", default="out_*.tar.gz",
                    help="which tarballs to consume. Narrow it (e.g. 'out_b500*.tar.gz') when "
                         "the submit directory holds more than one sweep -- the default would "
                         "otherwise merge a previous sweep's results into --stage as well, "
                         "and two sweeps at different budgets produce identical arm names.")
    ap.add_argument("--unpack-to", default="collected")
    ap.add_argument("--out", default="sweep_summary.csv")
    ap.add_argument("--stage", default=None,
                    help="also copy the seed trees here, ready to commit")
    ap.add_argument("--retry-list", default="joblist_retry.txt",
                    help="write the joblist lines for jobs that produced no result.csv, so "
                         "they can be resubmitted on their own")
    a = ap.parse_args()

    tars = sorted(glob.glob(str(pathlib.Path(a.indir) / a.pattern)))
    if not tars:
        sys.exit(f"no {a.pattern} under {a.indir}")
    dest = pathlib.Path(a.unpack_to)
    dest.mkdir(parents=True, exist_ok=True)
    for t in tars:
        with tarfile.open(t) as tf:
            tf.extractall(dest / pathlib.Path(t).name[: -len(".tar.gz")])
    print(f"unpacked {len(tars)} tarballs -> {dest}/")

    frames = []
    for f in sorted(dest.glob("*/results/seed_*/result.csv")):
        try:
            frames.append(pd.read_csv(f))
        except Exception as exc:
            print(f"  skipped {f}: {exc!r}")
    if not frames:
        sys.exit("no result.csv inside the tarballs -- did the jobs fail? check logs/")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(a.out, index=False)
    print(f"\n{len(df)} arm-runs over {df['seed'].nunique()} seeds -> {a.out}")
    if "error" in df.columns and df["error"].notna().any():
        bad = df[df["error"].notna()]
        print(f"\n!! {len(bad)} arm-runs errored:")
        print(bad[["seed", "arm", "error"]].to_string(index=False))
    # Which jobs came back with nothing usable. Worth knowing BEFORE reading the table: a
    # failure correlated with a swept axis biases whatever survived.
    jl = pathlib.Path("joblist.txt")
    if jl.exists():
        expected = [l.split(",")[1].strip() for l in jl.read_text().splitlines() if l.strip()]
        got = {pathlib.Path(t).name[len("out_"):-len(".tar.gz")] for t in tars}
        empty = [t for t in expected if t in got
                 and not list((dest / ("out_" + t)).glob("results/seed_*/result.csv"))]
        missing = [t for t in expected if t not in got]
        if empty:
            print("\n!! %d of %d jobs returned an EMPTY tarball:" % (len(empty), len(expected)))
            for t in empty:
                print("     %s   -> logs/job_%s.err" % (t, t))
        if missing:
            print("\n!! %d jobs never returned a tarball:" % len(missing))
            for t in missing:
                print("     %s   -> logs/job_%s.log" % (t, t))
        bad = set(empty) | set(missing)
        if bad and a.retry_list:
            keep = [l for l in jl.read_text().splitlines()
                    if l.strip() and l.split(",")[1].strip() in bad]
            pathlib.Path(a.retry_list).write_text("\n".join(keep) + "\n")
            print("\n-> wrote %s (%d jobs). Resubmit just those with:"
                  % (a.retry_list, len(keep)))
            print('     condor_submit sweep.sub -a "JOBLIST=%s"' % a.retry_list)

    print("\n%-26s%3s%13s%13s%13s%12s"
          % ("arm", "n", "infidelity", "diamond", "spam", "shots"))
    g = df.groupby("arm")
    tbl = g[["infidelity", "diamond", "spam"]].median()
    tbl["n"] = g.size()
    tbl["shots"] = g["accounted_revealed_shots"].median()
    for arm, r in tbl.sort_values("infidelity").iterrows():
        def cell(v):
            return ("%13.4e" % v) if pd.notna(v) else "%13s" % "--"
        sh = ("{:>12,.0f}".format(r["shots"]) if pd.notna(r["shots"])
              else "%12s" % "--")
        print("%-26s%3d%s%s%s%s" % (arm, int(r["n"]), cell(r["infidelity"]),
                                    cell(r["diamond"]), cell(r["spam"]), sh))

    if df["diamond"].isna().all():
        print("\n!! every diamond value is NaN -- cvxopt is missing on the EXECUTE nodes "
              "(build_env.sh only verifies the submit node).")

    if a.stage:
        out = pathlib.Path(a.stage)
        out.mkdir(parents=True, exist_ok=True)
        # Merge at the ARM level, not the seed level. Several jobs can contribute to one seed
        # (one per baseline fraction, say), each in its own tarball with its own
        # results/seed_XXXXXX/ tree. Replacing the seed directory would keep only the last
        # tarball's arms and silently drop the rest.
        n, arms = 0, 0
        for sd in sorted(dest.glob("*/results/seed_*")):
            tgt = out / sd.name
            tgt.mkdir(parents=True, exist_ok=True)
            n += 1
            for item in sorted(sd.iterdir()):
                if item.is_dir():
                    dst = tgt / item.name
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(item, dst)
                    arms += 1
                elif item.name == "result.csv":
                    # one job's rows; concatenate so the seed's result.csv covers every arm
                    prev = tgt / "result.csv"
                    new_rows = pd.read_csv(item)
                    if prev.exists():
                        new_rows = pd.concat([pd.read_csv(prev), new_rows], ignore_index=True)
                        new_rows = new_rows.drop_duplicates(subset=["seed", "arm"],
                                                            keep="last")
                    new_rows.to_csv(prev, index=False)
                else:
                    shutil.copy2(item, tgt / item.name)
        print(f"  merged {arms} arm directories")
        total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        print(f"\nstaged {n} seed trees -> {out}  ({total / 1048576:.1f} MB)")
        if total > 80 * 1048576:
            print("  WARNING: >80 MB. GitHub rejects single files over 100 MB and gets slow "
                  "past a few hundred MB of history. Consider committing only "
                  "sweep_summary.csv plus each arm's summary.json.")


if __name__ == "__main__":
    main()
