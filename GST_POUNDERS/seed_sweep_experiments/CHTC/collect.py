#!/usr/bin/env python
"""Unpack the per-job tarballs and merge every result.csv into sweep_summary.csv."""
import argparse, glob, tarfile, pathlib, sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--indir", default=".", help="where out_p*_s*.tar.gz landed")
ap.add_argument("--unpack-to", default="collected")
ap.add_argument("--out", default="sweep_summary.csv")
a = ap.parse_args()

dest = pathlib.Path(a.unpack_to); dest.mkdir(parents=True, exist_ok=True)
tars = sorted(glob.glob(str(pathlib.Path(a.indir) / "out_p*_s*.tar.gz")))
if not tars:
    sys.exit(f"no out_p*_s*.tar.gz under {a.indir}")
for t in tars:
    stem = pathlib.Path(t).stem.replace(".tar", "")
    with tarfile.open(t) as tf:
        tf.extractall(dest / stem)

rows = []
for f in sorted(dest.glob("*/results/seed_*/result.csv")):
    try:
        rows.append(pd.read_csv(f))
    except Exception as e:
        print("skip", f, e)
if not rows:
    sys.exit("no result.csv found inside the tarballs")
df = pd.concat(rows, ignore_index=True)
df.to_csv(a.out, index=False)

print(f"{len(tars)} tarballs -> {len(df)} rows -> {a.out}")
print(f"pilots: {sorted(df['pilot'].unique())}")
print(f"seeds : {df['seed'].nunique()}")
INF = "mean_gate_entanglement_infidelity_to_truth"
if INF in df.columns:
    print()
    print(df.pivot_table(index="pilot", columns="method", values=INF, aggfunc="median")
            .to_string(float_format=lambda v: f"{v:.3e}"))
