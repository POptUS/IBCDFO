# Overlay the POUNDERS / LM arms from all_methods_comparison onto the estimator_comparison plots.
#
# Usage from a notebook cell:
#     %run estimator_comparison_overlay.py
#     fig, tbl = overlay(EXPERIMENT_DIR)
#
# Read-only: loads summary.json / result JSONs and draws. Writes one PNG into
# estimator_comparison/ and nothing else. Never touches all_methods_comparison.
#
# WHY THE X-AXIS IS *TOTAL SHOTS* AND NOT SHOTS-PER-CIRCUIT
# ---------------------------------------------------------
# The estimator arms measure all 1918 circuits uniformly, so "shots per circuit" is well defined.
# adaptive_D and fixed_FPR measure a REVEALED SUBSET at varying depth (seed 101: 593 circuits,
# mean 717 shots), so there is no single shots-per-circuit for them. Total shots is the only
# quantity both sides agree on.
#
# Each POUNDERS/FPR arm is drawn at its ACCOUNTED total with a light bar extending to its
# PHYSICAL total. Those differ (seed 101 adaptive_D: 506,400 accounted vs 1,375,450 physical)
# because the harness simulates every circuit and then masks. Which one is the honest cost is a
# real open question for the paper -- an experimentalist only runs the revealed circuits, so
# accounted is arguably right, but no_FPR and LM have accounted == physical and had no such
# option. The bar makes that ambiguity visible instead of hiding it behind a choice.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INF = "mean_gate_entanglement_infidelity_to_truth"

# estimator_comparison arms (curves)
EST_ARMS = ["fwchi2", "chi2", "mle", "chi2_mle", "lgst"]
EST_LABEL = {"fwchi2": "freq-weighted chi2", "chi2": "model-prob chi2", "mle": "pure MLE",
             "chi2_mle": "chi2-seeded MLE (pyGSTi default)", "lgst": "LGST (linear inversion)"}
EST_COLOR = {"fwchi2": "#D55E00", "chi2": "#0072B2", "mle": "#009E73",
             "chi2_mle": "#555555", "lgst": "#E69F00"}
EST_MARKER = {"fwchi2": "o", "chi2": "s", "mle": "^", "chi2_mle": "*", "lgst": "D"}

# all_methods_comparison arms (points)
POU_FOLDERS = {"adaptive_D": "adaptive_D", "fixed_fpr": "fixed_FPR",
               "fixed_no_fpr": "no_FPR", "lm": "LM"}
POU_COLOR = {"adaptive_D": "#B00068", "fixed_FPR": "#7F3C00", "no_FPR": "#2E7D32", "LM": "#111111"}
POU_MARKER = {"adaptive_D": "P", "fixed_FPR": "X", "no_FPR": "v", "LM": "*"}


def load_estimator(results_dir):
    rows = []
    for p in sorted(Path(results_dir).glob("seed*_shots*_*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            pass
    if not rows:
        raise RuntimeError(f"no estimator results in {results_dir}")
    df = pd.DataFrame(rows)
    df = df[df.get("status", "ok") == "ok"].copy()
    df[INF] = pd.to_numeric(df[INF], errors="coerce")
    return df.dropna(subset=[INF])


def load_pounders(results_dir):
    rows = []
    for sd in sorted(Path(results_dir).glob("seed_*")):
        try:
            seed = int(sd.name.split("_")[1])
        except ValueError:
            continue
        for folder, label in POU_FOLDERS.items():
            f = sd / folder / "summary.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())
            if INF not in s:
                continue
            acc = float(s.get("accounted_revealed_shots", np.nan))
            phy = float(s.get("physical_precomputed_shots", acc))
            rows.append(dict(seed=seed, method=label, infidelity=float(s[INF]),
                             accounted=acc, physical=phy,
                             n_sigma=float(s.get("n_sigma", np.nan)),
                             revealed_circuits=s.get("revealed_circuits", np.nan),
                             mean_shots=s.get("mean_shots_per_circuit", np.nan)))
    if not rows:
        raise RuntimeError(f"no all_methods results in {results_dir}")
    return pd.DataFrame(rows)


def overlay(experiment_dir, match_seeds=True, save=True):
    """Draw estimator curves with the POUNDERS/LM arms overlaid.

    match_seeds=True restricts the POUNDERS arms to the seeds the estimator sweep has, so both
    sides describe the same realizations. Set False to use every seed on disk (more POUNDERS
    seeds, but then the two sides are not paired).
    """
    experiment_dir = Path(experiment_dir)
    est = load_estimator(experiment_dir / "estimator_comparison")
    pou = load_pounders(experiment_dir / "all_methods_comparison")

    est_seeds = sorted(est["seed"].unique())
    if match_seeds:
        pou = pou[pou["seed"].isin(est_seeds)]
        seed_note = f"seeds {est_seeds} (matched)"
    else:
        seed_note = f"estimator seeds {est_seeds}; POUNDERS seeds {sorted(pou['seed'].unique())} (UNMATCHED)"
    if pou.empty:
        raise RuntimeError("no POUNDERS arms left after seed matching -- pass match_seeds=False")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    # ---------------- left: infidelity vs total shots ----------------
    ax = axes[0]
    for arm in EST_ARMS:
        sub = est[est["arm"] == arm]
        if sub.empty:
            continue
        g = sub.groupby("total_shots")[INF]
        x = np.asarray(sorted(g.groups), dtype=float)
        med = g.median().reindex(x).to_numpy(float)
        lo = g.quantile(0.25).reindex(x).to_numpy(float)
        hi = g.quantile(0.75).reindex(x).to_numpy(float)
        ax.plot(x, med, color=EST_COLOR[arm], marker=EST_MARKER[arm], lw=2.0,
                label=EST_LABEL[arm], zorder=2)
        ax.fill_between(x, lo, hi, color=EST_COLOR[arm], alpha=0.13, zorder=1)

    for label, grp in pou.groupby("method"):
        acc = float(np.median(grp["accounted"]))
        phy = float(np.median(grp["physical"]))
        y = float(np.median(grp["infidelity"]))
        # accounted -> physical bar makes the budget ambiguity explicit
        if np.isfinite(phy) and phy > acc * 1.01:
            ax.plot([acc, phy], [y, y], color=POU_COLOR[label], lw=1.2, alpha=0.45,
                    ls="-", zorder=3)
            ax.plot([phy], [y], color=POU_COLOR[label], marker="|", ms=9, alpha=0.6, zorder=3)
        ax.scatter(grp["accounted"], grp["infidelity"], color=POU_COLOR[label], alpha=0.35,
                   marker=POU_MARKER[label], s=45, zorder=4)          # per-seed points
        ax.scatter([acc], [y], color=POU_COLOR[label], marker=POU_MARKER[label], s=190,
                   edgecolor="white", linewidth=1.3, zorder=5, label=f"{label} (POUNDERS/LM)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total shots  (accounted; bar extends to physically simulated)")
    ax.set_ylabel("mean gate entanglement infidelity to truth")
    ax.set_title(f"Estimators (curves, uniform over 1918 circuits) vs POUNDERS arms (points)\n{seed_note}",
                 fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")

    # ---------------- right: fit quality ----------------
    ax = axes[1]
    for arm in EST_ARMS:
        if arm == "lgst":
            continue     # its 2*deltaLogL is scored over all 1918 circuits incl. L=64 data it
                         # never used, so N_sigma is not a meaningful number for this arm
        sub = est[est["arm"] == arm]
        if sub.empty or "n_sigma" not in sub or sub["n_sigma"].isna().all():
            continue
        g = sub.groupby("total_shots")["n_sigma"]
        x = np.asarray(sorted(g.groups), dtype=float)
        ax.plot(x, g.median().reindex(x).to_numpy(float), color=EST_COLOR[arm],
                marker=EST_MARKER[arm], lw=2.0, label=EST_LABEL[arm])

    for label, grp in pou.groupby("method"):
        g = grp.dropna(subset=["n_sigma"])
        if g.empty:
            continue
        ax.scatter([np.median(g["accounted"])], [np.median(g["n_sigma"])],
                   color=POU_COLOR[label], marker=POU_MARKER[label], s=190,
                   edgecolor="white", linewidth=1.3, zorder=5, label=label)

    ax.axhline(1.0, color="black", ls="--", lw=1.0)
    ax.text(ax.get_xlim()[0], 1.05, " N_sigma = 1", fontsize=8, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total shots (accounted)")
    ax.set_ylabel("N_sigma  (2*deltaLogL vs dof)")
    ax.set_title("Fit quality  (LGST omitted: scored on circuits it never measured)", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5, ncol=2)

    fig.tight_layout()
    if save:
        out = experiment_dir / "estimator_comparison" / "estimator_vs_pounders.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("saved", out)

    # ---------------- comparison table ----------------
    tbl = (pou.groupby("method")
              .agg(seeds=("seed", "nunique"),
                   infidelity=("infidelity", "median"),
                   accounted=("accounted", "median"),
                   physical=("physical", "median"),
                   n_sigma=("n_sigma", "median"))
              .reset_index())
    est_tbl = (est[est["arm"] != "lgst"]
               .groupby(["arm", "total_shots"])[INF].median().reset_index())
    print("\nPOUNDERS / LM arms:")
    print(tbl.to_string(index=False,
                        float_format=lambda v: f"{v:,.4g}"))
    print("\nnearest estimator points by total shots:")
    for _, r in tbl.iterrows():
        near = est_tbl.iloc[(est_tbl["total_shots"] - r["accounted"]).abs().argsort()[:3]]
        best = near.sort_values(INF).iloc[0]
        print(f"  {r['method']:<11s} {r['infidelity']:.4e} @ {r['accounted']:>10,.0f} accounted"
              f"   |  best estimator near that budget: {best['arm']} "
              f"{best[INF]:.4e} @ {best['total_shots']:,.0f}")
    return fig, tbl


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    overlay(d)
    plt.show()
