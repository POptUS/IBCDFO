#!/usr/bin/env python
"""Answer the pilot-shot question from sweep_summary.csv.

The median table collect.py prints is not enough: adaptive_D is bimodal, so a
stable median is compatible with a large swing in the failure rate. This reports
the failure rate per pilot, the paired comparison against LM (same seed, same
budget -- so pairing is the right test), and whether any pilot is actually
distinguishable from the others.

    ./rolenv/bin/python analyze_sweep.py
    ./rolenv/bin/python analyze_sweep.py --csv sweep_summary.csv --ref LM
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

INF = "mean_gate_entanglement_infidelity_to_truth"
FAIL_FACTOR = 2.5          # same rule as analyze_all_methods.ipynb
FAIL_MAX_FRAC = 0.45


def failure_mask(v: np.ndarray) -> np.ndarray:
    """Median-relative rule, with the notebook's minority guard."""
    v = np.asarray(v, dtype=float)
    good = np.isfinite(v)
    if good.sum() < 3:
        return ~good
    med = np.median(v[good])
    m = good & (v > FAIL_FACTOR * med)
    if m.sum() > FAIL_MAX_FRAC * good.sum():   # not a minority -> not "failures"
        m = np.zeros_like(m)
    return m | ~good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sweep_summary.csv")
    ap.add_argument("--method", default="adaptive_D")
    ap.add_argument("--ref", default="LM")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    piv = df.pivot_table(index=["pilot", "seed"], columns="method", values=INF)
    if a.method not in piv or a.ref not in piv:
        raise SystemExit(f"need both {a.method} and {a.ref}; have {list(piv.columns)}")

    try:
        from scipy import stats
    except ImportError:
        stats = None

    # ---- 1. failure rate per pilot -------------------------------------------
    print(f"=== {a.method}: failure rate per pilot "
          f"(> {FAIL_FACTOR}x that pilot's median) ===")
    rows = []
    for pilot, g in piv.groupby(level="pilot"):
        v = g[a.method].to_numpy()
        m = failure_mask(v)
        clean = v[~m]
        rows.append({
            "pilot": pilot, "n": len(v), "failed": int(m.sum()),
            "fail_%": 100.0 * m.sum() / len(v),
            "median_all": np.median(v[np.isfinite(v)]),
            "median_ok": np.median(clean) if len(clean) else np.nan,
            "mean_ok": clean.mean() if len(clean) else np.nan,
            "worst": np.nanmax(v),
        })
    fr = pd.DataFrame(rows).set_index("pilot")
    print(fr.to_string(float_format=lambda x: f"{x:.3e}" if abs(x) < 1e-2 else f"{x:.1f}"))

    if stats is not None and fr["failed"].sum():
        tab = np.array([[int(r.failed), int(r.n - r.failed)] for r in fr.itertuples()])
        if (tab.sum(0) > 0).all():
            chi2, p, *_ = stats.chi2_contingency(tab)
            print(f"\n  chi2 test, failure rate constant across pilots: p = {p:.3f}"
                  f"  -> {'DIFFERS' if p < 0.05 else 'no evidence pilot changes it'}")
    elif not fr["failed"].sum():
        print("\n  no seeds flagged as failures at any pilot")

    # ---- 2. paired vs the reference ------------------------------------------
    print(f"\n=== {a.method} vs {a.ref}, paired by seed ===")
    rows = []
    for pilot, g in piv.groupby(level="pilot"):
        d = g[[a.method, a.ref]].dropna()
        ratio = (d[a.method] / d[a.ref]).to_numpy()
        r = {"pilot": pilot, "n": len(d), "median_ratio": np.median(ratio),
             "wins": int((ratio < 1).sum()), "geo_mean": float(np.exp(np.mean(np.log(ratio))))}
        if stats is not None and len(d) > 5:
            r["wilcoxon_p"] = stats.wilcoxon(d[a.method], d[a.ref]).pvalue
        rows.append(r)
    pr = pd.DataFrame(rows).set_index("pilot")
    print(pr.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n  median_ratio < 1 means {a.method} wins; wilcoxon_p < 0.05 means the")
    print(f"  difference at that pilot is not chance.")

    # ---- 3. does pilot matter at all? ----------------------------------------
    print(f"\n=== does pilot affect {a.method}? ===")
    groups = [g[a.method].dropna().to_numpy() for _, g in piv.groupby(level="pilot")]
    if len(groups) < 2:
        print(f"  only one pilot in this file -- nothing to compare across pilots")
        return
    if stats is not None:
        h, p = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis across all 6 pilots : p = {p:.3f}"
              f"  -> {'pilots DIFFER' if p < 0.05 else 'no detectable difference'}")
        flat = piv[a.method].dropna()
        rho, sp = stats.spearmanr(flat.index.get_level_values("pilot"), flat.to_numpy())
        print(f"  Spearman(pilot, infidelity)        : rho = {rho:+.3f}, p = {sp:.3f}")
        # the paired version: same seed at different pilots, so pair on seed
        best = pr["median_ratio"].idxmin()
        worst = pr["median_ratio"].idxmax()
        d = piv.loc[best, a.method].dropna()
        e = piv.loc[worst, a.method].dropna()
        common = d.index.intersection(e.index)
        if len(common) > 5:
            w = stats.wilcoxon(d.loc[common], e.loc[common]).pvalue
            print(f"  best pilot ({best}) vs worst ({worst}), paired on seed: p = {w:.3f}"
                  f"  -> {'real' if w < 0.05 else 'NOT distinguishable'}")
    else:
        print("  scipy unavailable; install it in the venv for the tests")


if __name__ == "__main__":
    main()
