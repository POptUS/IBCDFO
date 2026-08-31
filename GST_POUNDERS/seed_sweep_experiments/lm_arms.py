"""The pyGSTi Levenberg-Marquardt baseline arms, shared by the notebook and the CHTC jobs.

Three LM arms, differing in exactly one thing each so the pairs isolate one question:

    lm      full 1918-circuit design, pyGSTi's STANDARD recipe -- chi^2 up the max-length
            ladder with a single MLE at the end (objfn_builders left at None).
    lm_mle  full design, PURE MLE at every ladder stage.  lm vs lm_mle isolates the OBJECTIVE.
    lm_fpr  pure MLE on pyGSTi's built-in per-germ greedy FPR design -- fewer circuits, more
            shots each.  lm_mle vs lm_fpr isolates the DESIGN.

All three draw at the same data seed and the same accounted budget as the POUNDERS arms, so
every arm in a seed is one realisation of one experiment.

This module deliberately owns no policy: budgets, seeds and output directories are passed in.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

INF = "mean_gate_entanglement_infidelity_to_truth"
DD = "mean_gate_diamond_distance_to_truth"
SPAM = "mean_spam_vector_l2_error_to_truth"
SPAM_TD = "mean_spam_tracedist_to_truth"


# --------------------------------------------------------------------------- scoring


def score_and_write(prob, model, out_dir):
    """Truth-referenced metrics for a fitted model, plus the per-gate tables beside it.

    aligned_error_metrics aggregates diamond distance into its summary when
    report_diamond_distance is on, so infidelity, SPAM and diamond come from one call.
    """
    summ, gate_rows, spam_rows = prob.aligned_error_metrics(model, prob.truth_model, "truth")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gate_rows).to_csv(out_dir / "final_gate_errors.csv", index=False)
    pd.DataFrame(spam_rows).to_csv(out_dir / "final_spam_errors.csv", index=False)
    return summ


def stage_trajectory(prob, estimate, cfg):
    """Per-max-length-stage accuracy: one row per GST iteration of a pyGSTi estimate.

    The analysis notebooks overlay this on the POUNDERS per-iteration traces. Diamond is
    included because aligned_error_metrics computes it anyway when it is switched on.
    """
    mls = list(getattr(cfg, "max_lengths", []))
    keys = sorted([k for k in estimate.models if re.fullmatch(r"iteration \d+ estimate", k)],
                  key=lambda k: int(k.split()[1]))
    traj = []
    for si, k in enumerate(keys):
        try:
            st, _, _ = prob.aligned_error_metrics(estimate.models[k], prob.truth_model, "truth")
            traj.append({"stage": si, "max_length": (mls[si] if si < len(mls) else si),
                         INF: float(st[INF]), SPAM: float(st.get(SPAM, float("nan"))),
                         DD: float(st.get(DD, float("nan")))})
        except Exception:
            pass
    return traj


def _write_arm(prob, cfg, seed, out_dir, method, fit, estimate, shots, n_revealed, extra=None):
    """Persist one LM arm in the same shape run_one_experiment leaves a POUNDERS arm."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "x_best.npy", np.asarray(fit.to_vector(), dtype=float))
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
    (out_dir / "problem_metadata.json").write_text(json.dumps(
        {"data_seed": int(seed), "truth_seed": int(prob.truth_seed), "method": method,
         "num_parameters": int(prob.n), "num_circuits": len(prob.circuits)}, indent=2))
    summ = score_and_write(prob, fit, out_dir)
    out = {"data_seed": int(seed), "method": method,
           INF: float(summ[INF]),
           DD: float(summ.get(DD, float("nan"))),
           SPAM: float(summ.get(SPAM, float("nan"))),
           # aligned_error_metrics computes this alongside the L2; copy it through rather
           # than dropping it, so the LM arms carry the same operational SPAM metric as the
           # POUNDERS arms. Without it the notebooks can only plot SPAM for the POUNDERS
           # arms and the comparison has no baseline.
           SPAM_TD: float(summ.get(SPAM_TD, float("nan"))),
           "accounted_revealed_shots": int(shots.sum()),
           "physical_precomputed_shots": int(shots.sum()),
           "max_shots_per_circuit": int(shots.max()),
           "min_shots_per_circuit": int(shots.min()),
           "mean_shots_per_circuit": float(shots.mean()),
           "total_circuits": int(len(prob.circuits)),
           "revealed_circuits": int(n_revealed),
           "flag": method.upper()}
    out.update(extra or {})
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    if estimate is not None:
        pd.DataFrame(stage_trajectory(prob, estimate, cfg)).to_csv(
            out_dir / "lm_trajectory.csv", index=False)
    return out


def _optimizer(maxiter):
    from pygsti.optimize import SimplerLMOptimizer
    return SimplerLMOptimizer(maxiter=maxiter, maxfev=maxiter, tol=1e-6,
                              init_munu="auto", oob_action="reject")


# --------------------------------------------------------------------------- the arms


def fit_lm(prob, cfg, seed, target_shots, out_dir, maxiter=800, modes="CPTPLND",
           data_seed_offset=0):
    """pyGSTi's standard recipe: chi^2 per max-length stage, one MLE at the end."""
    import pygsti
    per = max(1, round(target_shots / len(prob.circuits)))
    shots = prob.normalize_shots(per)
    dataset = prob.simulate_dataset(shots, seed=data_seed_offset + seed)
    data = pygsti.protocols.ProtocolData(prob.design, dataset)
    res = pygsti.protocols.StandardGST(modes=modes, target_model=prob.target_model,
                                       optimizer=_optimizer(maxiter), verbosity=0).run(
                                       data, disable_checkpointing=True)
    keys = list(res.estimates.keys())
    est = res.estimates[modes if modes in res.estimates else keys[0]]
    fit = est.models["final iteration estimate"]
    return _write_arm(prob, cfg, seed, out_dir, "lm", fit, est, shots, len(prob.circuits),
                      extra={"objective": "chi2_ladder_then_mle"})


def _pure_mle(prob, design, dataset, maxiter, name):
    from pygsti.protocols import GST, GSTInitialModel, GSTObjFnBuilders, ProtocolData
    proto = GST(GSTInitialModel(prob.copy_model_at_x(np.zeros(prob.n))), gaugeopt_suite=None,
                objfn_builders=GSTObjFnBuilders.create_from(
                    "logl", always_perform_mle=True, only_perform_mle=True),
                optimizer=_optimizer(maxiter), badfit_options=None, verbosity=0, name=name)
    return proto.run(ProtocolData(design, dataset), disable_checkpointing=True).estimates[name]


def fit_lm_mle(prob, cfg, seed, target_shots, out_dir, maxiter=800, data_seed_offset=0):
    """Pure MLE at every ladder stage, on the FULL design."""
    per = max(1, round(target_shots / len(prob.circuits)))
    shots = prob.normalize_shots(per)
    dataset = prob.simulate_dataset(shots, seed=data_seed_offset + seed)
    est = _pure_mle(prob, prob.design, dataset, maxiter, "lmmle")
    fit = est.models["final iteration estimate"]
    return _write_arm(prob, cfg, seed, out_dir, "lm_mle", fit, est, shots, len(prob.circuits),
                      extra={"objective": "pure_mle"})


def builtin_fpr_design(prob, cfg, cache_dir, at="target", tol=10, fpr_seed=2024, log=print):
    """pyGSTi's per-germ greedy FPR design, cached on disk.

    The solve is deterministic given (model, tolerance, seed, pygsti version) and slow, so it
    is cached. At at="target" the model does not depend on the data seed, so one entry serves
    an entire sweep -- which also means every seed is reduced by the SAME design, as intended.
    """
    import hashlib
    import pygsti
    from pygsti.algorithms.fiducialpairreduction import (
        find_sufficient_fiducial_pairs_per_germ_greedy)
    from pygsti.protocols import StandardGSTDesign
    germs = list(prob.germs)
    key = dict(x=hashlib.sha256(np.ascontiguousarray(
                   np.asarray(prob.x0, float)).tobytes()).hexdigest()[:16],
               tol=tol, seed=fpr_seed, ngerms=len(germs),
               nprep=len(prob.prep_fiducials), nmeas=len(prob.meas_fiducials),
               pygsti=pygsti.__version__)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"pairs_{at}.json"
    pairs = None
    if f.exists():
        try:
            blob = json.loads(f.read_text())
            if blob.get("key") == key:
                pairs = {germs[int(i)]: [tuple(v) for v in vals]
                         for i, vals in blob["pairs"].items()}
                log(f"  FPR[{at}]: reused cached pair set")
        except Exception:
            pairs = None
    if pairs is None:
        pairs = find_sufficient_fiducial_pairs_per_germ_greedy(
            prob.copy_model_at_x(prob.x0), prob.prep_fiducials, prob.meas_fiducials, germs,
            constrain_to_tp=True, inv_trace_tol=tol, seed=fpr_seed, verbosity=0)
        f.write_text(json.dumps(dict(key=key, seconds=None, pairs={
            str(i): [[int(a), int(b)] for a, b in pairs.get(g, [])]
            for i, g in enumerate(germs)}), indent=2))
        log(f"  FPR[{at}]: solved")
    src = prob.processor_spec if prob.processor_spec is not None else prob.target_model
    design = StandardGSTDesign(src, prob.prep_fiducials, prob.meas_fiducials, germs,
                               list(cfg.max_lengths), fiducial_pairs=pairs)
    return design, list(design.all_circuits_needing_data)


def fit_lm_fpr(prob, cfg, seed, target_shots, out_dir, design_and_circuits,
               maxiter=800, data_seed_offset=0, fpr_at="target", tol=10, fpr_seed=2024):
    """Pure MLE on pyGSTi's built-in FPR design, at the SAME budget the other arms get.

    Same total shots as lm_mle, spread over fewer circuits, so each surviving circuit gets
    proportionally more. That makes lm_mle vs lm_fpr an exactly matched pair.
    """
    design, circs = design_and_circuits
    per = max(1, int(round(target_shots / len(circs))))
    vec = np.full(len(circs), per, dtype=int)
    dataset = prob.simulate_dataset(vec, seed=data_seed_offset + seed, circuits=circs)
    est = _pure_mle(prob, design, dataset, maxiter, "lmfpr")
    fit = est.models["final iteration estimate"]
    return _write_arm(prob, cfg, seed, out_dir, "lm_fpr", fit, est, vec, len(circs),
                      extra={"objective": "pure_mle", "fpr_at": fpr_at,
                             "fpr_tol": tol, "fpr_seed": fpr_seed})
