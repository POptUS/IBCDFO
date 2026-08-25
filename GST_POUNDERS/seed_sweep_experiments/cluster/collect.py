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
    ap.add_argument("--unpack-to", default="collected")
    ap.add_argument("--out", default="sweep_summary.csv")
    ap.add_argument("--stage", default=None,
                    help="also copy the seed trees here, ready to commit")
    a = ap.parse_args()

    tars = sorted(glob.glob(str(pathlib.Path(a.indir) / "out_*.tar.gz")))
    if not tars:
        sys.exit(f"no out_*.tar.gz under {a.indir}")
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
    print("\nmedian infidelity by arm:")
    print(df.groupby("arm")["infidelity"].median().sort_values().to_string())

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
