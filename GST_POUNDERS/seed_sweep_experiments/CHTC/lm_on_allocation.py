#!/usr/bin/env python
"""Run LM (pyGSTi StandardGST) on adaptive_D's shot ALLOCATION.

The question this answers: is adaptive_D's allocation good and its estimator the
bottleneck, or is the allocation itself adding nothing?

adaptive_D differs from LM in TWO ways at once -- where the shots go (D-optimal
reveal + top-ups vs uniform) and how the estimate is extracted (POUNDERS on
weighted least squares vs LM on the Poisson log-likelihood). The sweep results
cannot separate them. This script can: it takes the exact charged allocation
adaptive_D produced (final_shots_per_circuit.npy restricted to the revealed set,
reconstructed from fpr_selection_history.csv -- verified to reproduce the
accounted budget exactly), simulates a fresh dataset with those counts, and fits
it with the SAME LM protocol build_lm uses. Then, paired per seed:

    LM(adaptive allocation)  vs  LM(uniform allocation)   <- same estimator
                                                              only allocation differs

  * LM-on-adaptive WINS  -> the allocation has real value; the WLS/POUNDERS
                            estimator is what's throwing it away.
  * tie or LOSS          -> the D-criterion allocation adds nothing over uniform
                            even with a perfect estimator; fix the criterion,
                            not the optimizer.

Usage (on the AP, inside CHTC/):
    ./rolenv/bin/python lm_on_allocation.py by_pilot/pilot_0500
    ./rolenv/bin/python lm_on_allocation.py by_pilot/pilot_0500 --seeds 20001 20002
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

# make gst_seed_experiment importable whether run from CHTC/ (AP layout, where
# job.sh sets PYTHONPATH) or from anywhere else: the module lives one level up.
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import numpy as np
import pandas as pd

INF = "mean_gate_entanglement_infidelity_to_truth"


def revealed_mask(method_dir: Path, n_circuits: int) -> np.ndarray:
    """Union of every FPR/reveal selection round == the charged circuit set."""
    hist = pd.read_csv(method_dir / "fpr_selection_history.csv",
                       usecols=["selected_circuit_indices"])
    idx: set[int] = set()
    for s in hist["selected_circuit_indices"]:
        idx |= set(ast.literal_eval(s))
    mask = np.zeros(n_circuits, dtype=bool)
    mask[sorted(idx)] = True
    return mask


def fit_lm(prob, circuits, shots, seed, lm_maxiter):
    """The same LM protocol as run_one.build_lm, on an arbitrary allocation."""
    import pygsti
    from pygsti.optimize import SimplerLMOptimizer

    ds = prob.simulate_dataset(shots, seed=9_000_000 + seed, circuits=circuits)
    design = prob.design.truncate_to_available_data(ds)
    data = pygsti.protocols.ProtocolData(design, ds)
    opt = SimplerLMOptimizer(maxiter=lm_maxiter, maxfev=lm_maxiter, tol=1e-6,
                             init_munu="auto", oob_action="reject")
    proto = pygsti.protocols.StandardGST(modes="CPTPLND", target_model=prob.target_model,
                                         optimizer=opt, verbosity=0)
    res = proto.run(data)
    est = res.estimates["CPTPLND" if "CPTPLND" in res.estimates
                        else list(res.estimates)[0]]
    fit = est.models["final iteration estimate"]
    summ, _, _ = prob.aligned_error_metrics(fit, prob.truth_model, "truth")
    return float(summ[INF])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree", help="results tree, e.g. by_pilot/pilot_0500")
    ap.add_argument("--config", default="experiment_config.json")
    ap.add_argument("--method", default="adaptive_D")
    ap.add_argument("--lm-maxiter", type=int, default=800)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="subset of seeds (default: every seed_* dir)")
    ap.add_argument("--out", default="lm_on_allocation.csv")
    a = ap.parse_args()

    from gst_seed_experiment import ExperimentConfig, GSTProblem
    cfg = ExperimentConfig.from_json(a.config)

    rows = []
    for sd in sorted(Path(a.tree).glob("seed_*")):
        seed = int(sd.name.split("_")[1])
        if a.seeds and seed not in a.seeds:
            continue
        mdir = sd / a.method
        lm_summ = sd / "lm" / "summary.json"
        if not (mdir / "final_shots_per_circuit.npy").exists() or not lm_summ.exists():
            print(f"  {sd.name}: missing inputs, skipped")
            continue

        t0 = time.time()
        prob = GSTProblem(cfg, data_seed=seed)
        shots = np.load(mdir / "final_shots_per_circuit.npy").astype(int)
        mask = revealed_mask(mdir, len(prob.circuits))
        charged = int(shots[mask].sum())

        # sanity: the reconstruction must reproduce the accounted budget
        acc = json.loads((mdir / "summary.json").read_text()).get("accounted_revealed_shots")
        if acc is not None and int(acc) != charged:
            print(f"  {sd.name}: reconstructed {charged:,} != accounted {acc:,} -- skipped")
            continue

        kept = [c for c, m in zip(prob.circuits, mask) if m]
        inf_adapt = fit_lm(prob, kept, shots[mask], seed, a.lm_maxiter)

        ref = json.loads(lm_summ.read_text())
        rows.append({"seed": seed, "lm_on_adaptive_alloc": inf_adapt,
                     "lm_on_uniform": float(ref[INF]),
                     "alloc_shots": charged,
                     "uniform_shots": int(ref["accounted_revealed_shots"]),
                     "revealed": int(mask.sum()), "secs": round(time.time() - t0, 1)})
        r = rows[-1]
        print(f"  {sd.name}: LM(adaptive)={r['lm_on_adaptive_alloc']:.3e}  "
              f"LM(uniform)={r['lm_on_uniform']:.3e}  "
              f"ratio={r['lm_on_adaptive_alloc']/r['lm_on_uniform']:.3f}  "
              f"[{r['revealed']} circ, {charged:,} shots, {r['secs']}s]", flush=True)

    if not rows:
        raise SystemExit("no seeds processed")
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    r = df["lm_on_adaptive_alloc"] / df["lm_on_uniform"]
    print(f"\nn={len(df)}  median ratio LM(adaptive)/LM(uniform) = {r.median():.3f}  "
          f"adaptive-alloc wins {int((r < 1).sum())}/{len(df)}")
    if len(df) > 5:
        from scipy import stats
        p = stats.wilcoxon(df["lm_on_adaptive_alloc"], df["lm_on_uniform"]).pvalue
        print(f"Wilcoxon paired p = {p:.4f}")
    print(f"\n-> ratio < 1: the ALLOCATION carries value; the WLS estimator is the bottleneck.")
    print(f"-> ratio >= 1: the D-criterion allocation adds nothing; fix the criterion.")
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()
