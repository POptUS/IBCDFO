#!/usr/bin/env python
"""Unpack the per-job tarballs HTCondor returned and make them usable.

HTCondor transfers one ``out_p<pilot>_s<seed>.tar.gz`` back into the submit
directory per job. This script:

  1. unpacks them,
  2. writes ``sweep_summary.csv`` -- every method x pilot x seed in one table,
  3. rebuilds a per-pilot tree that ``analyze_all_methods.ipynb`` can open
     directly, i.e. ``by_pilot/pilot_0200/seed_020001/adaptive_D/...`` plus an
     ``all_methods_summary.csv`` in each, matching the layout the notebook's
     RESULTS_DIR expects.

Usage:
    python collect.py                       # unpack + summarise + build by_pilot/
    python collect.py --no-tree             # skip step 3 (saves disk)
    python collect.py --indir /path/to/out  # tarballs elsewhere
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import shutil
import sys
import tarfile

import pandas as pd

INF = "mean_gate_entanglement_infidelity_to_truth"
# folder name on disk -> the label the notebook uses
FOLDER_TO_LABEL = {"fixed_fpr": "fixed_FPR", "fixed_no_fpr": "no_FPR",
                   "adaptive_D": "adaptive_D", "adaptive_A": "adaptive_A",
                   "adaptive_L": "adaptive_L", "lm": "LM"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=".", help="where out_p*_s*.tar.gz landed")
    ap.add_argument("--unpack-to", default="collected")
    ap.add_argument("--out", default="sweep_summary.csv")
    ap.add_argument("--tree-dir", default="by_pilot")
    ap.add_argument("--no-tree", action="store_true",
                    help="skip the notebook-compatible per-pilot tree")
    a = ap.parse_args()

    tars = sorted(glob.glob(str(pathlib.Path(a.indir) / "out_p*_s*.tar.gz")))
    if not tars:
        sys.exit(f"no out_p*_s*.tar.gz under {a.indir}")

    dest = pathlib.Path(a.unpack_to)
    dest.mkdir(parents=True, exist_ok=True)
    for t in tars:
        stem = pathlib.Path(t).name[: -len(".tar.gz")]
        with tarfile.open(t) as tf:
            tf.extractall(dest / stem)
    print(f"unpacked {len(tars)} tarballs -> {dest}/")

    # ---- 1. the flat summary -------------------------------------------------
    rows = []
    for f in sorted(dest.glob("*/results/seed_*/result.csv")):
        try:
            rows.append(pd.read_csv(f))
        except Exception as e:
            print(f"  skip {f}: {e}")
    if not rows:
        sys.exit("no result.csv inside the tarballs -- did the jobs fail? check logs/")
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} rows -> {a.out}")

    # ---- 2. the per-pilot tree the notebook can read -------------------------
    if not a.no_tree:
        tree = pathlib.Path(a.tree_dir)
        made = 0
        for meta in sorted(dest.glob("*/results/seed_*/cell_meta.json")):
            info = json.loads(meta.read_text())
            pilot, seed = int(info["pilot"]), int(info["seed"])
            src = meta.parent
            dst = tree / f"pilot_{pilot:04d}" / f"seed_{seed:06d}"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            made += 1
        # per-pilot all_methods_summary.csv, exactly the columns the notebook wants
        for pdir in sorted(tree.glob("pilot_*")):
            p = int(pdir.name.split("_")[1])
            sub = df[df["pilot"] == p]
            if sub.empty:
                continue
            keep = sub.rename(columns={"seed": "seed", "method": "method"})[
                [c for c in ["seed", "method", INF, "accounted_revealed_shots",
                             "max_shots_per_circuit", "flag"] if c in sub.columns]]
            keep.to_csv(pdir / "all_methods_summary.csv", index=False)
        print(f"{made} cells -> {tree}/pilot_XXXX/seed_XXXXXX/  "
              f"({len(list(tree.glob('pilot_*')))} pilots)")
        print(f"   point the notebook's RESULTS_DIR at one of these, e.g.")
        first = next(iter(sorted(tree.glob("pilot_*"))), None)
        if first:
            print(f"   RESULTS_DIR = Path(r\"{first.resolve()}\")")

    # ---- 3. the answer to 'which pilot wins' ---------------------------------
    print()
    print(f"seeds per pilot: {df.groupby('pilot')['seed'].nunique().to_dict()}")
    if INF in df.columns:
        print()
        print("median infidelity to truth, by pilot x method:")
        print(df.pivot_table(index="pilot", columns="method", values=INF, aggfunc="median")
                .to_string(float_format=lambda v: f"{v:.3e}"))
        print()
        print("adaptive_D / LM ratio per pilot (below 1.0 = adaptive wins):")
        piv = df.pivot_table(index=["pilot", "seed"], columns="method", values=INF)
        if {"adaptive_D", "LM"} <= set(piv.columns):
            r = (piv["adaptive_D"] / piv["LM"]).groupby(level="pilot")
            print(pd.DataFrame({"median_ratio": r.median(),
                                "wins": r.apply(lambda s: int((s < 1).sum())),
                                "n": r.size()}).to_string(float_format=lambda v: f"{v:.3f}"))
    failed = df[df.get("flag").astype(str).eq("FAILED")] if "flag" in df.columns else df.iloc[:0]
    if len(failed):
        print(f"\n{len(failed)} FAILED cells:")
        print(failed[["pilot", "seed", "method"]].to_string(index=False))


if __name__ == "__main__":
    main()
